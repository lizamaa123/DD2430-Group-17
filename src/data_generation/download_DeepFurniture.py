import json
import os
import random
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Compact per-scene item lists (item IDs, labels, bounding boxes), built from
# every scene's annotation.json the first time duplicates are searched for.
# Reading ~24k annotation files takes a while, so later runs read this one
# file instead and only parse scenes that were added since.
SCENE_ITEMS_CACHE = os.path.join("data", "metadata", "scene_items.json")

# Scenes listed here (one scene_id per line) are treated as not existing:
# they are never extracted, sampled or displayed. It lives next to the code,
# not in data/, so it is committed and shared by everyone. Written by
# delete_duplicate_scenes.
EXCLUDED_SCENES_PATH = os.path.join(os.path.dirname(__file__), "deepfurniture_excluded_scenes.txt")

# Which file extensions to keep when extracting each archive type; anything
# else (e.g. a scene's depth map, which is a .png alongside the .jpg) is
# skipped during extraction so it's never written to disk.
KEEP_EXTENSIONS = {
    "furnitures": (".jpg", ".jpeg", ".png"),
    "scenes": (".jpg", ".jpeg", ".json"),
}


def _load_excluded_scenes():
    """
    Read the list of excluded scenes (EXCLUDED_SCENES_PATH).

    Args:
        None.

    Returns:
        Set of scene_ids to ignore. Empty if the file doesn't exist.
    """
    if not os.path.exists(EXCLUDED_SCENES_PATH):
        return set()
    with open(EXCLUDED_SCENES_PATH) as f:
        lines = (line.strip() for line in f)
        return {line for line in lines if line and not line.startswith("#")}


def _save_excluded_scenes(scene_ids):
    """
    Write the list of excluded scenes (EXCLUDED_SCENES_PATH), sorted so that
    git diffs stay small.

    Args:
        scene_ids: Iterable of scene_ids to exclude.

    Returns:
        None.
    """
    with open(EXCLUDED_SCENES_PATH, "w") as f:
        f.write("# Scenes that are ignored everywhere (duplicates of another scene).\n")
        f.write("# One scene_id per line. Written by delete_duplicate_scenes.\n")
        for scene_id in sorted(set(scene_ids)):
            f.write(f"{scene_id}\n")


def _download_and_extract_archive(subfolder, archive_name, keep_extensions, cache_dir):
    """
    Download one archive from data/{subfolder}/ in the DeepFurniture repo,
    extract only the members whose extension is in keep_extensions, then
    delete the downloaded archive. Scenes in EXCLUDED_SCENES_PATH are not
    extracted.

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

    # Scene archives hold one folder per scene: {scene_id}/image.jpg etc.
    excluded = _load_excluded_scenes() if subfolder == "scenes" else set()

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.split("/")[0] in excluded:
                continue
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
    excluded = _load_excluded_scenes()
    scene_index = [scene for scene in scene_index if scene["scene_id"] not in excluded]
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


def _compact_instances(annotation):
    """
    Keep only what duplicate checking and display need from a scene
    annotation: per instance, the item ID, its category and styles, and
    its bounding box in the room image. Style tags are deduplicated.

    Args:
        annotation: A loaded annotation.json dict.

    Returns:
        List of dicts with "furniture_id" (str), "category" (str),
        "styles" (list of str) and "box" ([xMin, xMax, yMin, yMax] in
        pixels), one per instance.
    """
    instances = []
    for instance in clean_scene_annotation(annotation)["instances"]:
        box = instance["boundingBox"]
        instances.append(
            {
                "furniture_id": str(instance["identityID"]),
                "category": instance["categoryName"],
                "styles": instance["styleNames"],
                "box": [box["xMin"], box["xMax"], box["yMin"], box["yMax"]],
            }
        )
    return instances


def _load_scene_items(cache_dir, num_workers=8):
    """
    Load the item list of every scene downloaded to cache_dir, using the
    SCENE_ITEMS_CACHE file where possible and parsing only the scenes it
    doesn't cover yet (the cache is then updated). Excluded scenes
    (EXCLUDED_SCENES_PATH) are left out.

    Only downloaded scenes are covered; nothing is fetched from the Hub,
    since that would mean downloading every scene archive (~42 GB).

    Args:
        cache_dir: Local root directory data is stored under.
        num_workers: Number of threads used to read annotation files
            (reading many small files is faster in parallel, especially
            when they have to be fetched from iCloud first).

    Returns:
        Dict mapping scene_id to its list of instances (see
        _compact_instances).
    """
    scenes_dir = os.path.join(cache_dir, "data", "scenes")
    if not os.path.isdir(scenes_dir):
        return {}
    excluded = _load_excluded_scenes()
    on_disk = sorted(
        scene_id
        for scene_id in os.listdir(scenes_dir)
        if scene_id not in excluded and os.path.exists(os.path.join(scenes_dir, scene_id, "annotation.json"))
    )

    cache_path = os.path.join(cache_dir, SCENE_ITEMS_CACHE)
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)

    to_read = [scene_id for scene_id in on_disk if scene_id not in cached]
    if to_read:

        def read_scene(scene_id):
            with open(os.path.join(scenes_dir, scene_id, "annotation.json")) as f:
                return scene_id, _compact_instances(json.load(f))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(executor.map(read_scene, to_read), total=len(to_read), desc="Reading scene annotations")
            )
        cached.update(results)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({scene_id: cached[scene_id] for scene_id in sorted(cached)}, f)

    return {scene_id: cached[scene_id] for scene_id in on_disk}


def _items_key(instances):
    """
    What a scene's items boil down to when checking for duplicates: its
    item IDs, sorted, with repeats kept (two identical chairs differ from
    one chair).

    Args:
        instances: A scene's list of instances (see _compact_instances).

    Returns:
        Tuple of furniture IDs.
    """
    return tuple(sorted(instance["furniture_id"] for instance in instances))


def _box_iou(box_a, box_b):
    """
    Intersection over union (IoU) of two bounding boxes: the area the two
    boxes share, divided by the total area covered by either of them
    (the shared part counted once). A standard measure of box overlap.

        IoU = area(A ∩ B) / area(A ∪ B)

    1 means the boxes are identical, 0 means they don't overlap at all.
    Example: two 10x10 boxes, the second shifted 2 pixels sideways, share
    80 px² out of 120 px² in total, so IoU = 80 / 120 = 0.67.

    Args:
        box_a, box_b: [xMin, xMax, yMin, yMax] in pixels.

    Returns:
        float in [0, 1].
    """
    x_overlap = max(0, min(box_a[1], box_b[1]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[2], box_b[2]))
    intersection = x_overlap * y_overlap
    area_a = (box_a[1] - box_a[0]) * (box_a[3] - box_a[2])
    area_b = (box_b[1] - box_b[0]) * (box_b[3] - box_b[2])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _view_similarity(instances_a, instances_b):
    """
    How similar the camera views of two scenes with the same items are:
    the mean IoU of the bounding boxes of corresponding items.

    The overlap of each pair of boxes is measured with IoU (intersection
    over union, see _box_iou) and averaged over the items.

    DeepFurniture has no 3D positions, so this is the closest thing to the
    "same layout" check used for InstructScene. Near 1 means (almost) the
    same image, e.g. the same render with the camera nudged slightly; low
    values mean the room is photographed from a different angle or the
    items are arranged differently. Instances of the same item (e.g. two
    identical chairs) are paired left to right.

    Args:
        instances_a, instances_b: Instance lists of two scenes with the
            same items (same _items_key).

    Returns:
        float in [0, 1].
    """
    boxes_a, boxes_b = defaultdict(list), defaultdict(list)
    for instance in instances_a:
        boxes_a[instance["furniture_id"]].append(instance["box"])
    for instance in instances_b:
        boxes_b[instance["furniture_id"]].append(instance["box"])

    ious = []
    for furniture_id, boxes in boxes_a.items():
        for box_a, box_b in zip(sorted(boxes), sorted(boxes_b[furniture_id])):
            ious.append(_box_iou(box_a, box_b))
    return sum(ious) / len(ious) if ious else 0.0


def _find_duplicate_groups(scenes, min_items):
    """
    Group scenes that contain the same items (same item IDs, same number
    of each). Scenes with fewer than min_items items are ignored.

    Args:
        scenes: Dict mapping scene_id to its list of instances.
        min_items: Scenes with fewer items than this are ignored.

    Returns:
        Tuple (checked, groups):
          - checked: dict of the scenes that were considered.
          - groups: list of lists of scene_ids (each sorted, at least two
            per list), ordered by the first scene_id of each list.
    """
    checked = {scene_id: instances for scene_id, instances in scenes.items() if len(instances) >= min_items}

    by_items = defaultdict(list)
    for scene_id in sorted(checked):
        by_items[_items_key(checked[scene_id])].append(scene_id)

    groups = [scene_ids for scene_ids in by_items.values() if len(scene_ids) >= 2]
    groups.sort(key=lambda scene_ids: scene_ids[0])
    return checked, groups


def find_and_show_duplicate_scenes(
    num_groups=3,
    scenes_per_group=4,
    min_items=3,
    same_view_iou=0.5,
    only_different_views=False,
    largest_first=False,
    num_workers=8,
    cache_dir=DATA_DIR,
    seed=None,
):
    """
    Find scenes that contain exactly the same items, display a few of
    those groups, and return how many there are. Nothing is deleted or
    changed (apart from writing the SCENE_ITEMS_CACHE file).

    The DeepFurniture counterpart of find_and_show_duplicate_rooms in
    download_InstructScene.py. Two scenes count as having the same items
    if they contain the same item IDs, with the same number of each.
    Scenes with fewer than min_items items are ignored, because small
    scenes (e.g. just one sofa) often match by coincidence.

    Each displayed group shows the scene images first, then the shared
    items. Whether the scenes in a group also show the same view is
    checked and reported: same items AND the items in the same places in
    the image is a stronger sign of a duplicate than the same items seen
    from another angle. Since DeepFurniture has no 3D positions, the view
    is compared through the items' 2D bounding boxes (see
    _view_similarity), using IoU (intersection over union: shared area
    divided by total area of two boxes, from 0 = no overlap to 1 =
    identical; see _box_iou).

    Only downloaded scenes are checked (see download_deepfurniture_subset).
    The first run reads every scene's annotation.json, which takes a few
    minutes; later runs use the SCENE_ITEMS_CACHE file. Images of scenes
    and items that aren't on disk are fetched from the Hub for display and
    deleted again afterward, as in visualize_random_scenes.

    Args:
        num_groups: How many duplicate groups to display.
        scenes_per_group: Max scenes to display per group (larger groups
            are cut off, but are counted in full in the return value).
        min_items: Scenes with fewer items than this are ignored.
        same_view_iou: Threshold for calling a group "same view", between
            0 and 1. IoU stands for intersection over union, a measure of
            how much two bounding boxes overlap (shared area divided by
            the area covered by either box; 1 = identical, 0 = no
            overlap). For each scene in a group, the IoU of every item's
            box is compared with that item's box in the group's first
            scene and averaged. The group counts as "same view" if every
            scene's average is at least this value; lower averages mean
            the same items are placed differently in the image (another
            camera angle or a rearranged room). Higher is stricter.
        only_different_views: If True, only consider groups whose scenes
            don't all show the same view (the same items photographed from
            another angle, or rearranged). Everything below (the counts
            and the displayed groups) then covers only those groups.
        largest_first: If True, display the groups with the most scenes
            instead of random ones (seed is then ignored).
        num_workers: Number of threads used when reading annotation files.
        cache_dir: Local root directory data is stored under. Defaults
            to data/deepfurniture/ at the repo root.
        seed: Optional random seed, for reproducible choice of groups.

    Returns:
        dict with:
          - "num_scenes_checked": scenes with at least min_items items.
          - "num_groups": groups of scenes with identical items.
          - "num_scenes_in_groups": scenes that are in some group.
          - "num_redundant_scenes": how many scenes would be left over if
            each group were reduced to a single scene (scenes in groups
            minus number of groups).
          - "num_groups_same_view": groups where every scene also shows
            the same view.
          - "groups": list of dicts, one per group, with "scene_ids",
            "min_view_iou" (lowest mean box IoU, i.e. intersection over
            union, of a scene with the group's first scene) and
            "same_view".
    """
    scenes = _load_scene_items(cache_dir, num_workers)
    if not scenes:
        raise FileNotFoundError(
            f"No downloaded scenes found under {os.path.join(cache_dir, 'data', 'scenes')}. "
            "Run download_deepfurniture_subset first."
        )

    checked, scene_id_groups = _find_duplicate_groups(scenes, min_items)

    groups = []
    for scene_ids in scene_id_groups:
        first = checked[scene_ids[0]]
        min_view_iou = min(_view_similarity(first, checked[scene_id]) for scene_id in scene_ids[1:])
        same_view = min_view_iou >= same_view_iou
        if only_different_views and same_view:
            continue
        groups.append({"scene_ids": scene_ids, "min_view_iou": min_view_iou, "same_view": same_view})

    summary = {
        "num_scenes_checked": len(checked),
        "num_groups": len(groups),
        "num_scenes_in_groups": sum(len(group["scene_ids"]) for group in groups),
        "num_redundant_scenes": sum(len(group["scene_ids"]) - 1 for group in groups),
        "num_groups_same_view": sum(group["same_view"] for group in groups),
        "groups": groups,
    }
    print(
        f"{summary['num_groups']} groups of scenes with identical items"
        f"{' and different views' if only_different_views else ''} "
        f"({summary['num_scenes_in_groups']} of {summary['num_scenes_checked']} scenes with at least "
        f"{min_items} items; in {summary['num_groups_same_view']} of the groups the view is the same too, "
        f"mean box IoU >= {same_view_iou})"
    )

    if largest_first:
        shown = sorted(groups, key=lambda group: len(group["scene_ids"]), reverse=True)[:num_groups]
    else:
        shown = random.Random(seed).sample(groups, k=min(num_groups, len(groups)))
    if not shown:
        return summary

    with open(_ensure_local_file("data/metadata/scene_index.json", cache_dir)) as f:
        scene_chunk_by_id = {s["scene_id"]: s["chunk_name"] for s in json.load(f)["scenes"]}
    with open(_ensure_local_file("data/metadata/furniture_index.json", cache_dir)) as f:
        furniture_chunk_by_id = {item["furniture_id"]: item["chunk_name"] for item in json.load(f)["furnitures"]}

    for number, group in enumerate(shown, start=1):
        group_scene_ids = group["scene_ids"][:scenes_per_group]
        first = checked[group_scene_ids[0]]

        images, labels = [], []
        cleanup = []  # files fetched only for this display, deleted afterward

        for scene_id in group_scene_ids:
            image_path, annotation_path, was_downloaded = _ensure_scene_files(
                scene_id, scene_chunk_by_id[scene_id], cache_dir
            )
            image = Image.open(image_path)
            image.load()
            images.append(image)
            if scene_id == group_scene_ids[0]:
                view_label = "reference view"
            else:
                view_label = f"box IoU with first: {_view_similarity(first, checked[scene_id]):.2f}"
            labels.append(f"Scene {scene_id[:12]}\n{view_label}")
            if was_downloaded:
                cleanup += [image_path, annotation_path]

        # One panel per distinct item, labelled with how often it appears.
        counts = defaultdict(int)
        info_by_id = {}
        for instance in first:
            counts[instance["furniture_id"]] += 1
            info_by_id.setdefault(instance["furniture_id"], instance)

        for furniture_id in sorted(counts):
            chunk_name = furniture_chunk_by_id.get(furniture_id)
            if chunk_name is None:
                continue  # not in the metadata index; skip rather than guess
            item_path, was_downloaded = _ensure_furniture_image(furniture_id, chunk_name, cache_dir)
            image = Image.open(item_path)
            image.load()
            if was_downloaded:
                cleanup.append(item_path)

            info = info_by_id[furniture_id]
            styles = ", ".join(info["styles"]) or "unknown style"
            count = f" x{counts[furniture_id]}" if counts[furniture_id] > 1 else ""
            images.append(image)
            labels.append(f"{info['category']}{count}\n({styles})")

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
        for ax in axes[num_panels:]:
            ax.axis("off")

        view = "same view" if group["same_view"] else "different views"
        cut_off = f", showing {len(group_scene_ids)}" if len(group["scene_ids"]) > len(group_scene_ids) else ""
        fig.suptitle(
            f"Group {number}: {len(group['scene_ids'])} scenes with identical items "
            f"({view}, min box IoU {group['min_view_iou']:.2f}{cut_off}). "
            f"Scenes first, then the shared items.",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()

        for path in cleanup:
            os.remove(path)
        for path in cleanup:
            folder = os.path.dirname(path)
            if os.path.basename(os.path.dirname(folder)) == "scenes" and os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)

    return summary


def delete_duplicate_scenes(
    min_items=3,
    dry_run=True,
    num_workers=8,
    cache_dir=DATA_DIR,
):
    """
    Remove duplicate scenes, keeping exactly one scene per group of scenes
    with the same items, and make sure the removed scenes never come back.

    The DeepFurniture counterpart of delete_duplicate_rooms in
    download_InstructScene.py. Duplicates are scenes with the same item IDs
    (same number of each), as in find_and_show_duplicate_scenes, whatever
    their view: scenes that show the same items from another angle (low
    bounding-box IoU, intersection over union) are removed too, since they
    are practically the same room. Scenes with fewer than min_items items
    are ignored, as in find_and_show_duplicate_scenes.

    The scene kept is the first one when sorted by scene_id. That is
    arbitrary but deterministic, so it's the same on every machine.

    With dry_run=True (the default) only prints and returns what would
    happen. Otherwise:
      1. The removed scene_ids are added to EXCLUDED_SCENES_PATH
         (src/data_generation/deepfurniture_excluded_scenes.txt). Commit
         this file so everyone ignores the same scenes. The download and
         visualization functions skip every scene in it.
      2. The removed scenes' folders (image.jpg and annotation.json) are
         deleted. Furniture images stay, since the kept scene uses the
         same items.

    Which scenes are duplicates is decided from the whole dataset, so the
    result is the same on every machine. Deleting therefore requires every
    scene in scene_index.json (apart from already excluded ones) to be
    downloaded; a dry run works on whatever is downloaded and says what's
    missing.

    Args:
        min_items: Scenes with fewer items than this are ignored.
        dry_run: If True, change nothing and only report.
        num_workers: Number of threads used when reading annotation files
            that aren't in the SCENE_ITEMS_CACHE file yet.
        cache_dir: Local root directory data is stored under. Defaults
            to data/deepfurniture/ at the repo root.

    Returns:
        dict with:
          - "dry_run": whether anything was changed.
          - "num_scenes_checked": scenes with at least min_items items.
          - "num_groups": groups of scenes with identical items.
          - "num_removed": scenes removed (or that would be, in a dry run).
          - "num_groups_different_view": groups whose scenes don't all show
            the same view (mean box IoU with the kept scene below 0.5).
          - "groups": list of dicts with "keep" and "remove" (list of
            scene_ids).
    """
    scenes = _load_scene_items(cache_dir, num_workers)
    if not scenes:
        raise FileNotFoundError(
            f"No downloaded scenes found under {os.path.join(cache_dir, 'data', 'scenes')}. "
            "Run download_deepfurniture_subset first."
        )

    with open(_ensure_local_file("data/metadata/scene_index.json", cache_dir)) as f:
        all_scene_ids = {scene["scene_id"] for scene in json.load(f)["scenes"]}
    missing = all_scene_ids - _load_excluded_scenes() - set(scenes)
    if missing:
        message = (
            f"{len(missing)} of the dataset's scenes aren't downloaded, so duplicates involving "
            "them can't be found."
        )
        if not dry_run:
            raise RuntimeError(
                f"{message} Download all scenes (download_deepfurniture_subset) before deleting, "
                "so the excluded list is the same on every machine."
            )
        print(f"{message} Showing what would happen for the downloaded scenes only.")

    checked, scene_id_groups = _find_duplicate_groups(scenes, min_items)
    groups = [{"keep": scene_ids[0], "remove": scene_ids[1:]} for scene_ids in scene_id_groups]
    to_remove = [scene_id for group in groups for scene_id in group["remove"]]

    num_different_view = sum(
        min(_view_similarity(checked[ids[0]], checked[scene_id]) for scene_id in ids[1:]) < 0.5
        for ids in scene_id_groups
    )

    summary = {
        "dry_run": dry_run,
        "num_scenes_checked": len(checked),
        "num_groups": len(groups),
        "num_removed": len(to_remove),
        "num_groups_different_view": num_different_view,
        "groups": groups,
    }

    verb = "Would remove" if dry_run else "Removing"
    print(
        f"{verb} {len(to_remove)} of {len(checked)} scenes with at least {min_items} items "
        f"({len(groups)} groups, one scene kept per group; {len(scenes) - len(checked)} scenes with "
        f"fewer items weren't checked)."
    )
    print(
        f"{num_different_view} of the groups have scenes with different views (mean box IoU < 0.5); "
        "these are removed too."
    )

    if dry_run or not to_remove:
        return summary

    # The exclusion list is written first, so the scenes stay out even if
    # something fails while deleting the local files.
    _save_excluded_scenes(_load_excluded_scenes() | set(to_remove))

    scenes_dir = os.path.join(cache_dir, "data", "scenes")
    for scene_id in tqdm(to_remove, desc="Deleting scenes"):
        scene_dir = os.path.join(scenes_dir, scene_id)
        if os.path.isdir(scene_dir):
            shutil.rmtree(scene_dir)

    print(f"Added {len(to_remove)} scenes to {os.path.basename(EXCLUDED_SCENES_PATH)}. Remember to commit it.")
    return summary


if __name__ == "__main__":
    #visualize_random_scenes()
    find_and_show_duplicate_scenes(num_groups=3)
