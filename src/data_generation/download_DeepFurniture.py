import json
import os
import random
import tarfile

from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt

REPO_ID = "byliu/DeepFurniture"

# Where downloaded/extracted files are cached, relative to the repo root
# (mirrors data_preprocessing.py, which also writes to data/).
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "deepfurniture")

NUM_FURNITURE_ARCHIVES = 25  # furnitures_0000.tar.gz .. furnitures_0024.tar.gz
NUM_SCENE_ARCHIVES = 25  # scenes_0000.tar.gz .. scenes_0024.tar.gz

METADATA_FILES = (
    "categories.json",
    "dataset_info.json",
    "furniture_index.json",
    "furnitures.jsonl",
    "query_index.json",
    "scene_index.json",
    "styles.json",
)

# Which file extensions to keep when extracting each archive type; anything
# else (e.g. a scene's depth map, which is a .png alongside the .jpg) is
# skipped during extraction so it's never written to disk.
KEEP_EXTENSIONS = {
    "furnitures": (".jpg", ".jpeg", ".png"),
    "scenes": (".jpg", ".jpeg", ".json"),
}


def _download_and_extract_archive(subfolder, archive_name, keep_extensions, cache_dir):
    """
    Download one archive from data/{subfolder}/ in the DeepFurniture repo,
    extract only the members whose extension is in keep_extensions, then
    delete the downloaded archive.

    Args:
        subfolder: Dataset folder the archive lives in on the Hub
            ("furnitures" or "scenes").
        archive_name: Filename of the archive, e.g. "scenes_0007.tar.gz".
        keep_extensions: Tuple of lowercase file extensions to keep from
            inside the archive; every other member is skipped.
        cache_dir: Local root directory; extracted files are written to
            cache_dir/data/{subfolder}/ (e.g. cache_dir/data/scenes/).

    Returns:
        None.
    """
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/{subfolder}/{archive_name}",
        local_dir=cache_dir,
    )

    extract_dir = os.path.dirname(archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(keep_extensions):
                tar.extract(member, path=extract_dir)

    # The archive itself isn't needed once the wanted files are extracted.
    os.remove(archive_path)


def _deduplicate_instance_styles(instance):
    """
    Remove duplicate style tags from a single scene instance. The raw
    annotation format sometimes repeats the same style ID/name pair
    (e.g. styleIDs [4, 4]) - most likely because multiple annotators
    tagged the same instance and their agreeing votes were appended
    rather than deduplicated - so this keeps only the first occurrence
    of each style ID.

    Args:
        instance: One entry from an annotation.json "instances" list,
            with parallel "styleIDs" and "styleNames" lists.

    Returns:
        A copy of instance with "styleIDs" and "styleNames" deduplicated
        (order-preserving); every other field is left unchanged.
    """
    style_ids = instance.get("styleIDs", [])
    style_names = instance.get("styleNames", [])

    seen_ids = set()
    deduped_ids = []
    deduped_names = []
    for style_id, style_name in zip(style_ids, style_names):
        if style_id in seen_ids:
            continue
        seen_ids.add(style_id)
        deduped_ids.append(style_id)
        deduped_names.append(style_name)

    cleaned_instance = dict(instance)
    cleaned_instance["styleIDs"] = deduped_ids
    cleaned_instance["styleNames"] = deduped_names
    return cleaned_instance


def clean_scene_annotation(annotation):
    """
    Deduplicate style tags across every instance in a loaded scene
    annotation. Pure function on an in-memory dict - safe to call
    whether that dict was read from a file on disk or fetched on the fly
    and never saved, so callers get clean data either way.

    Args:
        annotation: A loaded annotation.json dict, with an "instances"
            list (each instance having "styleIDs"/"styleNames").

    Returns:
        A new annotation dict with each instance's style tags
        deduplicated.
    """
    cleaned_annotation = dict(annotation)
    cleaned_annotation["instances"] = [
        _deduplicate_instance_styles(instance) for instance in annotation.get("instances", [])
    ]
    return cleaned_annotation


def clean_downloaded_furnitures_metadata(cache_dir=DATA_DIR):
    """
    Rewrite metadata/furnitures.jsonl with duplicate style_ids removed
    from each entry.

    This is the actual source of the duplicate style tags: each
    furniture item has ONE style_ids list in furnitures.jsonl, and every
    scene's annotation.json instance just copies that same list verbatim
    wherever the item appears - so cleaning this one ~1.6 MB, ~24.7k-line
    file fixes it everywhere, rather than needing to touch each of the
    ~24k scene annotation files individually.

    Run this once against data you downloaded before this cleanup
    existed. Anything downloaded from now on via
    download_deepfurniture_subset is already cleaned as part of that
    function.

    Args:
        cache_dir: Local root directory data is stored under. Defaults
            to data/deepfurniture/ at the repo root.

    Returns:
        int: number of furniture entries that had duplicate style_ids
        removed.
    """
    furnitures_path = os.path.join(cache_dir, "data", "metadata", "furnitures.jsonl")

    entries = []
    num_cleaned = 0
    with open(furnitures_path) as f:
        for line in f:
            entry = json.loads(line)
            style_ids = entry.get("style_ids")
            if style_ids is not None:
                deduped = list(dict.fromkeys(style_ids))
                if deduped != style_ids:
                    num_cleaned += 1
                entry["style_ids"] = deduped
            entries.append(entry)

    with open(furnitures_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return num_cleaned


def download_deepfurniture_subset(
    include_scenes=True,
    num_furniture_archives=NUM_FURNITURE_ARCHIVES,
    num_scene_archives=NUM_SCENE_ARCHIVES,
    cache_dir=DATA_DIR,
):
    """
    OBS: Takes about 10 minutes to run and 20 GB of storage. 
    
    Download the parts of the DeepFurniture dataset needed for room-aware
    furniture recommendation, skipping everything else.

    Downloaded:
      - data/furnitures/*  -> one preview image per furniture item.
      - data/scenes/*      -> one RGB room image + annotation JSON per
                              scene (which furniture items appear together
                              in each room), with depth maps dropped.
      - data/metadata/*    -> categories.json, styles.json, and the index
                              files tying items/scenes/queries together
                              (~9 MB total, always fetched in full).

    Skipped entirely:
      - data/queries/*  -> cropped furniture instances re-derivable from
                            scenes + furnitures; not needed here.
      - depth maps      -> the .png depth file bundled alongside each
                            scene's .jpg is discarded right after that
                            scene's archive is extracted.

    Archives are processed one at a time (download -> extract wanted
    files -> delete archive), so the full ~43 GB is never on disk at once.

    Args:
        include_scenes: Whether to download scene archives at all (they're
            the large part, ~42 GB before filtering). Set False to fetch
            only item images + metadata.
        num_furniture_archives: How many of the 25 furnitures_XXXX.tar.gz
            archives to download (0 skips item images entirely).
        num_scene_archives: How many of the 25 scenes_XXXX.tar.gz archives
            to download; ignored if include_scenes is False.
        cache_dir: Local directory everything is downloaded/extracted
            into. Defaults to data/deepfurniture/ at the repo root.

    Returns:
        None.
    """
    os.makedirs(cache_dir, exist_ok=True)

    for filename in METADATA_FILES:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/metadata/{filename}",
            local_dir=cache_dir,
        )

    # Deduplicate style tags at the source (see
    # clean_downloaded_furnitures_metadata for why they show up at all).
    clean_downloaded_furnitures_metadata(cache_dir)

    for i in range(num_furniture_archives):
        archive_name = f"furnitures_{i:04d}.tar.gz"
        _download_and_extract_archive("furnitures", archive_name, KEEP_EXTENSIONS["furnitures"], cache_dir)

    if include_scenes:
        for i in range(num_scene_archives):
            archive_name = f"scenes_{i:04d}.tar.gz"
            _download_and_extract_archive("scenes", archive_name, KEEP_EXTENSIONS["scenes"], cache_dir)


def _ensure_local_file(repo_relative_path, cache_dir):
    """
    Return the local path to a single file from the DeepFurniture repo,
    using it in place if it's already on disk and only reaching out to
    the Hub otherwise.

    Args:
        repo_relative_path: Path of the file inside the dataset repo,
            e.g. "data/metadata/categories.json".
        cache_dir: Local root directory data is stored under.

    Returns:
        Local filesystem path (str) to the file.
    """
    local_path = os.path.join(cache_dir, repo_relative_path)
    if os.path.exists(local_path):
        return local_path

    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=repo_relative_path,
        local_dir=cache_dir,
    )


def _ensure_scene_files(scene_id, chunk_name, cache_dir):
    """
    Return local paths to one scene's image and annotation JSON, fetching
    just that scene - not the whole chunk archive it belongs to - if it
    isn't already on disk.

    Args:
        scene_id: The scene's ID, e.g. "DVD3HFDYEJI4OKYQBT3WKSY8".
        chunk_name: Name of the archive the scene lives in on the Hub,
            e.g. "scenes_0000" (from scene_index.json).
        cache_dir: Local root directory data is stored under.

    Returns:
        An (image_path, annotation_path, was_downloaded) tuple.
        was_downloaded is True if these files had to be fetched just now,
        so the caller knows to clean them up again afterward rather than
        leaving a single stray scene behind that was never asked for.
    """
    scene_dir = os.path.join(cache_dir, "data", "scenes", scene_id)
    image_path = os.path.join(scene_dir, "image.jpg")
    annotation_path = os.path.join(scene_dir, "annotation.json")

    if os.path.exists(image_path) and os.path.exists(annotation_path):
        return image_path, annotation_path, False

    # Downloaded to the Hub's own cache (not cache_dir) since we only want
    # to keep the one scene we extract from it, not the whole archive.
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/scenes/{chunk_name}.tar.gz",
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(f"{scene_id}/")]
        tar.extractall(path=os.path.join(cache_dir, "data", "scenes"), members=members)

    return image_path, annotation_path, True


def _ensure_furniture_image(furniture_id, chunk_name, cache_dir):
    """
    Return the local path to one furniture item's image, fetching just
    that item - not the whole chunk archive it belongs to - if it isn't
    already on disk.

    Args:
        furniture_id: The furniture item's ID, e.g. "2880315".
        chunk_name: Name of the archive the item lives in on the Hub,
            e.g. "furnitures_0012" (from furniture_index.json).
        cache_dir: Local root directory data is stored under.

    Returns:
        An (image_path, was_downloaded) tuple. was_downloaded is True if
        the file had to be fetched just now, so the caller knows whether
        to clean it up again afterward.
    """
    image_path = os.path.join(cache_dir, "data", "furnitures", f"{furniture_id}.jpg")
    if os.path.exists(image_path):
        return image_path, False

    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/furnitures/{chunk_name}.tar.gz",
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember(f"{furniture_id}.jpg")
        tar.extract(member, path=os.path.join(cache_dir, "data", "furnitures"))

    return image_path, True


def visualize_random_scenes(num_scenes=3, cache_dir=DATA_DIR, seed=None):
    """
    Sanity-check the downloaded/extracted data by displaying random scenes
    (room images) next to every furniture item annotated in that room,
    with each image's category and style labels shown beneath it.

    Works whether or not the data is already downloaded: anything already
    present under cache_dir is used as-is; anything missing is fetched
    just for that one scene/item and deleted again once displayed, so
    running this against an empty cache_dir doesn't leave a partial
    dataset behind, and running it against a fully downloaded cache_dir
    never re-downloads anything.

    This is a visualization/debugging aid for checking that the data
    extraction is correct - it isn't part of the data pipeline itself.

    Args:
        num_scenes: How many random scenes to sample and display.
        cache_dir: Local root directory to look for/download data in.
            Defaults to data/deepfurniture/ at the repo root. Point this
            at an empty directory to exercise the "nothing downloaded
            yet" path instead of whatever is already cached there.
        seed: Optional random seed, for reproducible sampling.

    Returns:
        None.
    """
    rng = random.Random(seed)

    with open(_ensure_local_file("data/metadata/scene_index.json", cache_dir)) as f:
        scene_index = json.load(f)["scenes"]
    with open(_ensure_local_file("data/metadata/furniture_index.json", cache_dir)) as f:
        furniture_index = json.load(f)["furnitures"]
    furniture_chunk_by_id = {f["furniture_id"]: f["chunk_name"] for f in furniture_index}

    sampled_scenes = rng.sample(scene_index, k=min(num_scenes, len(scene_index)))

    for scene_entry in sampled_scenes:
        scene_id = scene_entry["scene_id"]

        image_path, annotation_path, scene_was_downloaded = _ensure_scene_files(
            scene_id, scene_entry["chunk_name"], cache_dir
        )
        with open(annotation_path) as f:
            annotation = clean_scene_annotation(json.load(f))
        room_image = Image.open(image_path)
        room_image.load()

        item_images = []
        item_labels = []
        item_cleanup = []
        for instance in annotation["instances"]:
            furniture_id = str(instance["identityID"])
            chunk_name = furniture_chunk_by_id.get(furniture_id)
            if chunk_name is None:
                continue  # not in the metadata index; skip rather than guess

            item_path, item_was_downloaded = _ensure_furniture_image(furniture_id, chunk_name, cache_dir)
            item_image = Image.open(item_path)
            item_image.load()

            item_images.append(item_image)
            styles = ", ".join(instance["styleNames"]) or "unknown style"
            item_labels.append(f"{instance['categoryName']}\n({styles})")
            item_cleanup.append((item_path, item_was_downloaded))

        images = [room_image] + item_images
        labels = ["Room"] + item_labels

        max_cols = 7
        num_panels = len(images)
        num_cols = min(max_cols, num_panels)
        num_rows = -(-num_panels // num_cols)  # ceil division

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        axes = [axes] if num_panels == 1 else axes.flatten()

        for ax, image, label in zip(axes, images, labels):
            ax.imshow(image)
            ax.set_title(label)
            ax.axis("off")

        # Hide any unused panels left over in the last row, e.g. a 7-item
        # room fills a 2x5 grid but only uses 8 of its 10 slots.
        for ax in axes[num_panels:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        # Clean up anything that was only fetched for this check, so a
        # visualization run against an empty cache_dir leaves it empty
        # again afterward.
        if scene_was_downloaded:
            os.remove(image_path)
            os.remove(annotation_path)
            os.rmdir(os.path.dirname(image_path))
        for item_path, item_was_downloaded in item_cleanup:
            if item_was_downloaded:
                os.remove(item_path)


if __name__ == "__main__":
    visualize_random_scenes()
