import io
import json
import math
import os
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO_ID = "byliu/DeepFurniture"

# Where downloaded/extracted files are cached, relative to the repo root
# (mirrors data_preprocessing.py, which also writes to data/).
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "deepfurniture")

# Scratch cache for testing the "nothing downloaded yet" path. Sits next to
# DATA_DIR so any leftover files are easy to spot; it should stay empty.
EMPTY_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "empty_deepfurniture")

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

# How much is fetched per network request when streaming an archive from the
# Hub without downloading it (see _stream_archive_members).
REMOTE_BLOCK_SIZE = 8 * 1024 * 1024

# Scenes listed here (one scene_id per line) are treated as not existing:
# they are never extracted, sampled or displayed. It lives next to the code,
# not in data/, so it is committed and shared by everyone. Written by
# delete_duplicate_scenes.
EXCLUDED_SCENES_PATH = os.path.join(os.path.dirname(__file__), "deepfurniture_excluded_scenes.txt")

# Scratch folder (inside cache_dir) that archives are downloaded into before
# extraction. It is deleted when an archive is done or the run fails, and
# swept at the start of every bulk download in case the previous run was
# killed outright (e.g. VS Code's stop button sends SIGKILL, so no cleanup
# code gets a chance to run).
TMP_DOWNLOAD_DIRNAME = ".tmp_download"

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


def _remove_leftover_downloads(cache_dir):
    """
    Delete the scratch download folder (TMP_DOWNLOAD_DIRNAME) left behind
    by an earlier run that was killed before it could clean up after
    itself.

    Args:
        cache_dir: Local root directory data is stored under.

    Returns:
        None.
    """
    shutil.rmtree(os.path.join(cache_dir, TMP_DOWNLOAD_DIRNAME), ignore_errors=True)


def _download_and_extract_archive(subfolder, archive_name, keep_extensions, cache_dir):
    """
    Download one archive from data/{subfolder}/ in the DeepFurniture repo,
    extract only the members whose extension is in keep_extensions, then
    delete the downloaded archive. Scenes in EXCLUDED_SCENES_PATH are not
    extracted.

    The archive is downloaded into a scratch folder (TMP_DOWNLOAD_DIRNAME)
    that is deleted afterwards whether or not the download/extraction
    finished, so an interrupted run (Ctrl-C or an error) doesn't leave a
    partial multi-GB archive behind. If the process is killed outright,
    the folder is removed at the start of the next download instead.

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
    tmp_dir = os.path.join(cache_dir, TMP_DOWNLOAD_DIRNAME)
    extract_dir = os.path.join(cache_dir, "data", subfolder)

    # Scene archives hold one folder per scene: {scene_id}/image.jpg etc.
    excluded = _load_excluded_scenes() if subfolder == "scenes" else set()

    try:
        archive_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/{subfolder}/{archive_name}",
            local_dir=tmp_dir,
        )
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.split("/")[0] in excluded:
                    continue
                if member.isfile() and member.name.lower().endswith(keep_extensions):
                    tar.extract(member, path=extract_dir)
    finally:
        # The archive (and any half-finished download of it) isn't needed
        # once the wanted files are extracted - or once the run is aborted.
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    _remove_leftover_downloads(cache_dir)

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


def _read_metadata_json(filename, cache_dir):
    """
    Load one metadata JSON file (e.g. "scene_index.json"): from
    cache_dir/data/metadata/ if it's downloaded, otherwise straight from
    the Hub into memory, without saving anything.

    Args:
        filename: Name of the file in data/metadata/ in the dataset repo.
        cache_dir: Local root directory data is stored under.

    Returns:
        The parsed JSON.
    """
    local_path = os.path.join(cache_dir, "data", "metadata", filename)
    if os.path.exists(local_path):
        with open(local_path) as f:
            return json.load(f)
    with HfFileSystem().open(f"datasets/{REPO_ID}/data/metadata/{filename}", "rb") as f:
        return json.load(f)


def _stream_archive_members(subfolder, chunk_name, wanted, max_members=None):
    """
    Read selected files out of one archive on the Hub without downloading
    it to disk: the archive is streamed and decompressed in memory, and
    only the wanted members are kept.

    A .tar.gz can't be read at random positions, so everything up to the
    last wanted member is still transferred over the network (each archive
    is roughly 1-2 GB). Reading stops as soon as max_members have been
    found.

    Args:
        subfolder: "scenes" or "furnitures".
        chunk_name: Archive name without extension, e.g. "scenes_0003".
        wanted: Function taking a member name (e.g. "SCENEID/image.jpg")
            and returning a key to store it under, or None to skip it.
        max_members: Stop after this many wanted members (None reads the
            whole archive).

    Returns:
        Dict mapping each key returned by wanted to that member's bytes.
    """
    found = {}
    path = f"datasets/{REPO_ID}/data/{subfolder}/{chunk_name}.tar.gz"
    with HfFileSystem().open(path, "rb", block_size=REMOTE_BLOCK_SIZE) as f:
        with tarfile.open(fileobj=f, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key = wanted(member.name)
                if key is None:
                    continue
                found[key] = tar.extractfile(member).read()
                if max_members is not None and len(found) >= max_members:
                    break
    return found


def _read_scene_items_from_hub(num_workers=8, num_archives=None):
    """
    Read every scene's item list (see _compact_instances) straight from
    the scene archives on the Hub into memory. Nothing is saved, but all
    scene archives are streamed (~42 GB over the network), so this is slow;
    it exists so the analysis functions also work with no data downloaded.
    Excluded scenes (EXCLUDED_SCENES_PATH) are left out.

    Args:
        num_workers: Number of archives streamed in parallel.
        num_archives: Only read this many of the scene archives (each
            holds ~1,000 scenes), e.g. 1 for a quick test. None reads all.

    Returns:
        Dict mapping scene_id to its list of instances.
    """
    chunk_names = [f"scenes_{i:04d}" for i in range(NUM_SCENE_ARCHIVES)]
    if num_archives is not None:
        chunk_names = chunk_names[:num_archives]
    excluded = _load_excluded_scenes()

    def wanted(name):
        scene_id, _, filename = name.partition("/")
        return scene_id if filename == "annotation.json" and scene_id not in excluded else None

    def read_archive(chunk_name):
        members = _stream_archive_members("scenes", chunk_name, wanted)
        return {scene_id: _compact_instances(json.loads(data)) for scene_id, data in members.items()}

    print(f"Streaming {len(chunk_names)} scene archives from the Hub (nothing is saved; this takes a while).")
    scenes = {}
    with ThreadPoolExecutor(max_workers=max(1, min(num_workers, len(chunk_names)))) as executor:
        for result in tqdm(executor.map(read_archive, chunk_names), total=len(chunk_names), desc="Scene archives"):
            scenes.update(result)
    return dict(sorted(scenes.items()))


def _load_display_images(scene_ids, furniture_ids, cache_dir):
    """
    Load the images needed for a plot. Images that are downloaded are read
    from disk; the rest are read from the Hub into memory, streaming each
    archive involved only once and saving nothing.

    Args:
        scene_ids: Scenes whose room image is needed.
        furniture_ids: Items whose image is needed.
        cache_dir: Local root directory data is stored under.

    Returns:
        Tuple (scene_images, furniture_images): dicts mapping IDs to PIL
        images. IDs whose image can't be found are left out.
    """
    scene_images, furniture_images = {}, {}
    missing_scenes, missing_items = [], []

    for scene_id in scene_ids:
        path = os.path.join(cache_dir, "data", "scenes", scene_id, "image.jpg")
        if os.path.exists(path):
            scene_images[scene_id] = Image.open(path)
        else:
            missing_scenes.append(scene_id)
    for furniture_id in furniture_ids:
        path = os.path.join(cache_dir, "data", "furnitures", f"{furniture_id}.jpg")
        if os.path.exists(path):
            furniture_images[furniture_id] = Image.open(path)
        else:
            missing_items.append(furniture_id)

    if missing_scenes:
        chunk_by_id = {s["scene_id"]: s["chunk_name"] for s in _read_metadata_json("scene_index.json", cache_dir)["scenes"]}
        by_chunk = defaultdict(set)
        for scene_id in missing_scenes:
            if scene_id in chunk_by_id:
                by_chunk[chunk_by_id[scene_id]].add(scene_id)
        for chunk_name, ids in sorted(by_chunk.items()):
            print(f"Reading {len(ids)} scene image(s) from {chunk_name} on the Hub...")
            wanted = lambda name, ids=ids: name.split("/")[0] if name.endswith("/image.jpg") and name.split("/")[0] in ids else None
            for scene_id, data in _stream_archive_members("scenes", chunk_name, wanted, len(ids)).items():
                scene_images[scene_id] = Image.open(io.BytesIO(data))

    if missing_items:
        chunk_by_id = {
            f["furniture_id"]: f["chunk_name"] for f in _read_metadata_json("furniture_index.json", cache_dir)["furnitures"]
        }
        by_chunk = defaultdict(set)
        for furniture_id in missing_items:
            if furniture_id in chunk_by_id:
                by_chunk[chunk_by_id[furniture_id]].add(furniture_id)
        for chunk_name, ids in sorted(by_chunk.items()):
            print(f"Reading {len(ids)} item image(s) from {chunk_name} on the Hub...")
            wanted = lambda name, ids=ids: name[: -len(".jpg")] if name.endswith(".jpg") and name[: -len(".jpg")] in ids else None
            for furniture_id, data in _stream_archive_members("furnitures", chunk_name, wanted, len(ids)).items():
                furniture_images[furniture_id] = Image.open(io.BytesIO(data))

    for image in list(scene_images.values()) + list(furniture_images.values()):
        image.load()
    return scene_images, furniture_images


def _show_panels(images, labels, title, max_cols=7):
    """
    Show images in a grid (at most max_cols per row), each with its label
    above it, and a title over the whole figure.

    Args:
        images: List of PIL images.
        labels: List of str, one per image.
        title: Figure title.
        max_cols: Max images per row.

    Returns:
        None.
    """
    num_panels = len(images)
    num_cols = min(max_cols, num_panels)
    num_rows = -(-num_panels // num_cols)  # ceil division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = [axes] if num_panels == 1 else axes.flatten()
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    # Hide any unused panels left over in the last row.
    for ax in axes[num_panels:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def _load_scene(scene_id, chunk_name, cache_dir):
    """
    Load one scene's annotation and room image: from disk if the scene is
    downloaded, otherwise straight from its archive on the Hub into memory
    (one streamed pass for both files), without saving anything.

    Args:
        scene_id: The scene's ID, e.g. "DVD3HFDYEJI4OKYQBT3WKSY8".
        chunk_name: Name of the archive the scene lives in on the Hub,
            e.g. "scenes_0000" (from scene_index.json).
        cache_dir: Local root directory data is stored under.

    Returns:
        Tuple (annotation, image): the parsed annotation JSON and a loaded
        PIL image.
    """
    scene_dir = os.path.join(cache_dir, "data", "scenes", scene_id)
    image_path = os.path.join(scene_dir, "image.jpg")
    annotation_path = os.path.join(scene_dir, "annotation.json")

    if os.path.exists(image_path) and os.path.exists(annotation_path):
        with open(annotation_path) as f:
            annotation = json.load(f)
        image = Image.open(image_path)
    else:
        print(f"Reading scene {scene_id} from {chunk_name} on the Hub...")
        filenames = ("image.jpg", "annotation.json")

        def wanted(name):
            prefix, _, filename = name.partition("/")
            return filename if prefix == scene_id and filename in filenames else None

        members = _stream_archive_members("scenes", chunk_name, wanted, len(filenames))
        annotation = json.loads(members["annotation.json"])
        image = Image.open(io.BytesIO(members["image.jpg"]))

    image.load()
    return annotation, image


def _stream_first_scenes(chunk_name, num_scenes, excluded):
    """
    Read the first num_scenes scenes (annotation + room image) of one scene
    archive on the Hub into memory, without saving anything. Only the start
    of the archive is transferred - roughly 2 MB per scene - instead of
    reading up to an arbitrary scene's position in the ~2 GB archive.

    Args:
        chunk_name: Archive name without extension, e.g. "scenes_0003".
        num_scenes: How many complete scenes to read before stopping.
        excluded: Set of scene_ids to skip (see EXCLUDED_SCENES_PATH).

    Returns:
        Dict mapping scene_id to an (annotation, image) tuple: the parsed
        annotation JSON and a loaded PIL image. Fewer than num_scenes
        entries only if the archive runs out first.
    """
    filenames = ("image.jpg", "annotation.json")
    partial, complete = {}, {}
    path = f"datasets/{REPO_ID}/data/scenes/{chunk_name}.tar.gz"
    with HfFileSystem().open(path, "rb", block_size=REMOTE_BLOCK_SIZE) as f:
        with tarfile.open(fileobj=f, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                scene_id, _, filename = member.name.partition("/")
                if scene_id in excluded or filename not in filenames:
                    continue
                files = partial.setdefault(scene_id, {})
                files[filename] = tar.extractfile(member).read()
                if len(files) == len(filenames):
                    complete[scene_id] = partial.pop(scene_id)
                    if len(complete) >= num_scenes:
                        break

    scenes = {}
    for scene_id, files in complete.items():
        image = Image.open(io.BytesIO(files["image.jpg"]))
        image.load()
        scenes[scene_id] = (json.loads(files["annotation.json"]), image)
    return scenes


def visualize_random_scenes(num_scenes=3, cache_dir=DATA_DIR, seed=None, sample_from_first_n=None):
    """
    Sanity-check the downloaded/extracted data by displaying random scenes
    (room images) next to every furniture item annotated in that room,
    with each image's category and style labels shown above it.

    Works whether or not the data is already downloaded: anything present
    under cache_dir is used as-is; anything missing (metadata index files,
    the scene, item images) is read straight from the Hub into memory.
    Nothing is ever written to disk, so there is nothing to clean up - even
    if the run is interrupted or killed - and running it against an empty
    cache_dir never creates that folder.

    This is a visualization/debugging aid for checking that the data
    extraction is correct - it isn't part of the data pipeline itself.

    Args:
        num_scenes: How many random scenes to sample and display.
        cache_dir: Local root directory to look for data in. Defaults to
            data/deepfurniture/ at the repo root. Point this at an empty
            directory to exercise the "nothing downloaded yet" path
            instead of whatever is already cached there.
        seed: Optional random seed, for reproducible sampling.
        sample_from_first_n: Speed-up for when the data isn't downloaded.
            None (default) samples from all scenes, and a scene that isn't
            on disk can take several minutes to fetch, since its ~2 GB
            archive has to be read up to that scene's position. If set to
            a number N, scenes are instead sampled from the first N scenes
            of one randomly chosen archive, so only the start of that
            archive is read (~2 MB per scene; e.g. 10 reads ~20 MB). The
            sample is then less random: always early scenes, all from the
            same archive. The scenes' item images still come from the
            furniture archives (~35 MB each, often several per scene), so
            expect seconds to a couple of minutes rather than instant.
            Scenes are always read from the Hub in this mode; with the data
            downloaded, leave it at None, which is already instant.

    Returns:
        None.
    """
    rng = random.Random(seed)
    excluded = _load_excluded_scenes()

    if sample_from_first_n is None:
        scene_index = [
            scene for scene in _read_metadata_json("scene_index.json", cache_dir)["scenes"]
            if scene["scene_id"] not in excluded
        ]
        sampled = rng.sample(scene_index, k=min(num_scenes, len(scene_index)))
        # A generator, so each scene is fetched only right before it's shown.
        scenes = ((entry["scene_id"], *_load_scene(entry["scene_id"], entry["chunk_name"], cache_dir)) for entry in sampled)
    else:
        if sample_from_first_n < 1:
            raise ValueError(f"sample_from_first_n must be at least 1, got {sample_from_first_n}")
        chunk_name = rng.choice([f"scenes_{i:04d}" for i in range(NUM_SCENE_ARCHIVES)])
        print(f"Reading the first {sample_from_first_n} scenes of {chunk_name} from the Hub...")
        first_scenes = _stream_first_scenes(chunk_name, sample_from_first_n, excluded)
        sampled_ids = rng.sample(sorted(first_scenes), k=min(num_scenes, len(first_scenes)))
        scenes = ((scene_id, *first_scenes[scene_id]) for scene_id in sampled_ids)

    for scene_id, annotation, room_image in scenes:
        annotation = clean_scene_annotation(annotation)

        instances = annotation["instances"]
        furniture_ids = sorted({str(instance["identityID"]) for instance in instances})
        # Items not in the metadata index or without an image are left out
        # of the returned dict, and skipped below rather than guessed.
        _, furniture_images = _load_display_images([], furniture_ids, cache_dir)

        images = [room_image]
        labels = ["Room"]
        for instance in instances:
            furniture_id = str(instance["identityID"])
            if furniture_id not in furniture_images:
                continue
            images.append(furniture_images[furniture_id])
            styles = ", ".join(instance["styleNames"]) or "unknown style"
            labels.append(f"{instance['categoryName']}\n({styles})")

        _show_panels(images, labels, title=f"Scene {scene_id}")


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


def _load_local_scene_items(cache_dir, num_workers=8):
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


def _load_scene_items(cache_dir, use_local=True, num_workers=8, hub_scene_archives=None):
    """
    Get every scene's item list, from downloaded data if there is any (and
    use_local is True), otherwise straight from the Hub into memory
    (_read_scene_items_from_hub; slow, nothing is saved). Excluded scenes
    are left out either way.

    Args:
        cache_dir: Local root directory data is stored under.
        use_local: Whether to use downloaded data when there is some. Set
            False to test the no-download path even with data on disk.
        num_workers: Threads for reading annotation files or archives.
        hub_scene_archives: When reading from the Hub, only read this many
            scene archives (~1,000 scenes each), e.g. 1 for a quick test.
            None reads all. Ignored for downloaded data.

    Returns:
        Dict mapping scene_id to its list of instances (see
        _compact_instances).
    """
    if use_local:
        scenes = _load_local_scene_items(cache_dir, num_workers)
        if scenes:
            return scenes
    return _read_scene_items_from_hub(num_workers, hub_scene_archives)


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
    use_local=True,
    hub_scene_archives=None,
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

    Works whether or not the data is downloaded. With downloaded scenes,
    the first run reads every scene's annotation.json, which takes a while;
    later runs use the SCENE_ITEMS_CACHE file. With nothing downloaded (or
    use_local=False), every scene is read from the Hub into memory, which
    is slow (all scene archives are streamed) but saves nothing. Images
    that aren't on disk are read from the Hub into memory for display.

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
        use_local: Whether to use downloaded data when there is some. Set
            False to read everything from the Hub.
        hub_scene_archives: When reading from the Hub, only read this many
            of the 25 scene archives (~1,000 scenes each), e.g. 1 for a
            quick test. None reads all.
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
    scenes = _load_scene_items(cache_dir, use_local, num_workers, hub_scene_archives)
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

    for number, group in enumerate(shown, start=1):
        group_scene_ids = group["scene_ids"][:scenes_per_group]
        first = checked[group_scene_ids[0]]

        # One panel per distinct item, labelled with how often it appears.
        counts = Counter(instance["furniture_id"] for instance in first)
        info_by_id = {instance["furniture_id"]: instance for instance in first}
        scene_images, furniture_images = _load_display_images(group_scene_ids, sorted(counts), cache_dir)

        images, labels = [], []
        for scene_id in group_scene_ids:
            if scene_id not in scene_images:
                continue
            if scene_id == group_scene_ids[0]:
                view_label = "reference view"
            else:
                view_label = f"box IoU with first: {_view_similarity(first, checked[scene_id]):.2f}"
            images.append(scene_images[scene_id])
            labels.append(f"Scene {scene_id[:12]}\n{view_label}")

        for furniture_id in sorted(counts):
            if furniture_id not in furniture_images:
                continue  # no image for this item; skip rather than guess
            info = info_by_id[furniture_id]
            styles = ", ".join(info["styles"]) or "unknown style"
            count = f" x{counts[furniture_id]}" if counts[furniture_id] > 1 else ""
            images.append(furniture_images[furniture_id])
            labels.append(f"{info['category']}{count}\n({styles})")

        view = "same view" if group["same_view"] else "different views"
        cut_off = f", showing {len(group_scene_ids)}" if len(group["scene_ids"]) > len(group_scene_ids) else ""
        _show_panels(
            images,
            labels,
            f"Group {number}: {len(group['scene_ids'])} scenes with identical items "
            f"({view}, min box IoU {group['min_view_iou']:.2f}{cut_off}). "
            f"Scenes first, then the shared items.",
        )

    return summary


def delete_duplicate_scenes(
    min_items=3,
    dry_run=True,
    use_local=True,
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
    result is the same on every machine: downloaded data is only used if
    it covers every scene in scene_index.json (apart from already excluded
    ones); otherwise every scene is read from the Hub into memory (slow,
    nothing is saved), and only the exclusion list is updated for scenes
    that aren't downloaded.

    Args:
        min_items: Scenes with fewer items than this are ignored.
        dry_run: If True, change nothing and only report.
        use_local: Whether to use downloaded data when it covers every
            scene. Set False to always read from the Hub.
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
    all_scene_ids = {scene["scene_id"] for scene in _read_metadata_json("scene_index.json", cache_dir)["scenes"]}
    expected = all_scene_ids - _load_excluded_scenes()

    scenes = _load_local_scene_items(cache_dir, num_workers) if use_local else {}
    if not (scenes and expected <= set(scenes)):
        if scenes:
            print(
                f"{len(expected - set(scenes))} scenes aren't downloaded, so reading all scenes "
                "from the Hub instead."
            )
        scenes = _read_scene_items_from_hub(num_workers)

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


def _item_overlap(items_a, items_b):
    """
    How much two scenes' items overlap: the number of distinct items they
    share divided by the number of distinct items in either scene. This is
    the same intersection-over-union idea as the box IoU (see _box_iou),
    applied to sets of item IDs instead of areas (also called the Jaccard
    index). How often an item repeats within a scene is ignored.

        overlap = |A ∩ B| / |A ∪ B|

    1 means the same set of items, 0 means no items in common. Example:
    scenes with 8 and 7 distinct items that share 6 overlap
    6 / (8 + 7 - 6) = 0.67.

    Args:
        items_a, items_b: Sets of furniture IDs.

    Returns:
        float in [0, 1].
    """
    union = len(items_a | items_b)
    return len(items_a & items_b) / union if union else 0.0


def _find_similar_pairs(scenes, min_overlap, min_items):
    """
    Find every pair of scenes whose item overlap (see _item_overlap) is at
    least min_overlap. Scenes with fewer than min_items items are ignored.

    Comparing all ~190 million pairs directly would be slow, so this uses
    prefix filtering (Bayardo et al., "Scaling up all pairs similarity
    search", WWW 2007): with each scene's items sorted from rarest to most
    common, two scenes can only reach the threshold if they share an item
    among the first |A| - ceil(min_overlap * |A|) + 1 of either scene. Only
    such candidate pairs are compared, and the result is exact.

    Args:
        scenes: Dict mapping scene_id to its list of instances.
        min_overlap: Threshold in (0, 1].
        min_items: Scenes with fewer items (instances) than this are ignored.

    Returns:
        List of (overlap, scene_id_a, scene_id_b) tuples, scene_id_a <
        scene_id_b.
    """
    if not 0 < min_overlap <= 1:
        raise ValueError(f"min_overlap must be in (0, 1], got {min_overlap}")

    item_sets = {
        scene_id: frozenset(instance["furniture_id"] for instance in instances)
        for scene_id, instances in scenes.items()
        if len(instances) >= min_items
    }
    frequency = Counter(item for items in item_sets.values() for item in items)

    index = defaultdict(list)  # item -> scenes having it in their prefix
    candidates = set()
    for scene_id in sorted(item_sets):
        items = sorted(item_sets[scene_id], key=lambda item: (frequency[item], item))
        prefix_length = len(items) - math.ceil(min_overlap * len(items) - 1e-9) + 1
        for item in items[:prefix_length]:
            for other in index[item]:
                candidates.add((other, scene_id))
            index[item].append(scene_id)

    pairs = []
    for scene_a, scene_b in candidates:
        overlap = _item_overlap(item_sets[scene_a], item_sets[scene_b])
        if overlap >= min_overlap:
            pairs.append((overlap, scene_a, scene_b))
    pairs.sort(key=lambda pair: (pair[1], pair[2]))
    return pairs


def _connected_groups(pairs):
    """
    Turn pairs of similar scenes into groups: two scenes end up in the same
    group if they are linked by a chain of similar pairs (connected
    components). These are the groups to keep on one side of a train/test
    split.

    Args:
        pairs: Iterable of (overlap, scene_id_a, scene_id_b).

    Returns:
        List of sorted lists of scene_ids, largest group first.
    """
    parent = {}

    def find(scene_id):
        parent.setdefault(scene_id, scene_id)
        while parent[scene_id] != scene_id:
            parent[scene_id] = parent[parent[scene_id]]
            scene_id = parent[scene_id]
        return scene_id

    for _, scene_a, scene_b in pairs:
        root_a, root_b = find(scene_a), find(scene_b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    groups = defaultdict(list)
    for scene_id in parent:
        groups[find(scene_id)].append(scene_id)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: (-len(group), group[0]))


def find_and_show_similar_scenes(
    num_pairs=3,
    min_overlap=0.7,
    max_overlap=1.0,
    min_items=3,
    use_local=True,
    hub_scene_archives=None,
    num_workers=8,
    cache_dir=DATA_DIR,
    seed=None,
):
    """
    Find near-identical scenes: pairs that share most, but not necessarily
    all, of their items, e.g. the same room seen from another angle (some
    items are out of view) or with one item swapped. Display a few pairs
    and return how many there are. Nothing is deleted or changed.

    Exact duplicates are handled by find_and_show_duplicate_scenes and
    delete_duplicate_scenes; this catches what they miss. These scenes are
    real data, so rather than deleting them they should be kept on the same
    side of a train/test split; the returned "groups" are meant for that.

    Similarity is the item overlap (see _item_overlap): distinct items
    shared divided by distinct items in either scene, from 0 (nothing in
    common) to 1 (the same set of items).

    Each displayed pair shows the two scene images, then their items in
    three blocks: shared by both, only in the first scene, only in the
    second.

    Works whether or not the data is downloaded, like
    find_and_show_duplicate_scenes.

    Args:
        num_pairs: How many random pairs to display.
        min_overlap: Lowest item overlap that counts as similar, in (0, 1].
        max_overlap: Only display pairs with at most this overlap. Use it
            to look at a band, e.g. 0.5 to 0.6, when choosing a threshold.
            The counts and groups ignore it.
        min_items: Scenes with fewer items than this are ignored, because
            small scenes overlap a lot by coincidence.
        use_local: Whether to use downloaded data when there is some. Set
            False to read everything from the Hub.
        hub_scene_archives: When reading from the Hub, only read this many
            of the 25 scene archives, e.g. 1 for a quick test. None reads
            all.
        num_workers: Number of threads used when reading annotation files.
        cache_dir: Local root directory data is stored under. Defaults
            to data/deepfurniture/ at the repo root.
        seed: Optional random seed, for reproducible choice of pairs.

    Returns:
        dict with:
          - "num_scenes_checked": scenes with at least min_items items.
          - "num_pairs": pairs with overlap >= min_overlap.
          - "num_scenes_in_pairs": scenes that are similar to at least
            one other scene.
          - "groups": lists of scene_ids linked by chains of similar
            pairs, largest first (see _connected_groups).
          - "pairs": list of (overlap, scene_id_a, scene_id_b).
    """
    scenes = _load_scene_items(cache_dir, use_local, num_workers, hub_scene_archives)
    pairs = _find_similar_pairs(scenes, min_overlap, min_items)
    groups = _connected_groups(pairs)
    num_checked = sum(len(instances) >= min_items for instances in scenes.values())

    summary = {
        "num_scenes_checked": num_checked,
        "num_pairs": len(pairs),
        "num_scenes_in_pairs": sum(len(group) for group in groups),
        "groups": groups,
        "pairs": pairs,
    }
    print(
        f"{summary['num_pairs']} pairs of scenes with item overlap >= {min_overlap}; "
        f"{summary['num_scenes_in_pairs']} of {num_checked} scenes with at least {min_items} items "
        f"are in one, forming {len(groups)} groups (largest: {len(groups[0]) if groups else 0} scenes)."
    )

    in_band = [pair for pair in pairs if pair[0] <= max_overlap]
    shown = random.Random(seed).sample(in_band, k=min(num_pairs, len(in_band)))
    if not shown:
        return summary

    for number, (overlap, scene_a, scene_b) in enumerate(shown, start=1):
        info_by_id = {instance["furniture_id"]: instance for instance in scenes[scene_a] + scenes[scene_b]}
        items_a = {instance["furniture_id"] for instance in scenes[scene_a]}
        items_b = {instance["furniture_id"] for instance in scenes[scene_b]}
        blocks = [
            ("both", sorted(items_a & items_b)),
            ("only in A", sorted(items_a - items_b)),
            ("only in B", sorted(items_b - items_a)),
        ]
        scene_images, furniture_images = _load_display_images([scene_a, scene_b], sorted(items_a | items_b), cache_dir)

        images, labels = [], []
        for name, scene_id in (("A", scene_a), ("B", scene_b)):
            if scene_id in scene_images:
                images.append(scene_images[scene_id])
                labels.append(f"Scene {name}: {scene_id[:12]}\n{len(scenes[scene_id])} items")
        for block, furniture_ids in blocks:
            for furniture_id in furniture_ids:
                if furniture_id not in furniture_images:
                    continue
                info = info_by_id[furniture_id]
                images.append(furniture_images[furniture_id])
                labels.append(f"[{block}] {info['category']}\n({', '.join(info['styles']) or 'unknown style'})")

        _show_panels(
            images,
            labels,
            f"Pair {number}: item overlap {overlap:.2f} ({len(blocks[0][1])} shared, "
            f"{len(blocks[1][1])} only in A, {len(blocks[2][1])} only in B). Scenes first, then the items.",
        )

    return summary


def show_dataset_statistics(
    min_items=3,
    use_local=True,
    hub_scene_archives=None,
    num_workers=8,
    cache_dir=DATA_DIR,
):
    """
    Print and plot summary statistics of the scenes (excluded scenes left
    out), to see what the data looks like before modelling:

      - scene size: how many scenes have each number of items,
      - items per category,
      - item popularity: in how many scenes each item appears (most items
        appear in very few scenes - the long tail),
      - style labels per item instance, including how many have none,
      - near-identical scenes: how many scenes share at least 50 / 70 / 90 %
        of their items with another scene (see find_and_show_similar_scenes).

    Item counts per scene count every instance (two identical chairs are 2
    items), as in the annotations.

    Works whether or not the data is downloaded, like
    find_and_show_duplicate_scenes.

    Args:
        min_items: Scenes with fewer items than this are left out of the
            near-identical counts (as in find_and_show_similar_scenes); all
            other statistics cover every scene.
        use_local: Whether to use downloaded data when there is some. Set
            False to read everything from the Hub.
        hub_scene_archives: When reading from the Hub, only read this many
            of the 25 scene archives, e.g. 1 for a quick test. None reads
            all.
        num_workers: Number of threads used when reading annotation files.
        cache_dir: Local root directory data is stored under. Defaults
            to data/deepfurniture/ at the repo root.

    Returns:
        dict with the numbers behind the plots: "num_scenes",
        "num_instances", "num_distinct_items", "scene_sizes" (size -> number
        of scenes), "instances_per_category", "scenes_per_item_buckets",
        "instances_per_style" (an instance with several styles counts once
        per style; "(none)" counts instances without one),
        "num_scenes_with_repeated_item", "most_common_items" (top 10, as
        (furniture_id, category, number of scenes)) and
        "near_identical_scenes" (overlap threshold -> number of scenes).
    """
    scenes = _load_scene_items(cache_dir, use_local, num_workers, hub_scene_archives)
    instances = [instance for scene in scenes.values() for instance in scene]

    scene_sizes = Counter(len(scene) for scene in scenes.values())
    per_category = Counter(instance["category"] for instance in instances)
    scenes_per_item = Counter(
        furniture_id for scene in scenes.values() for furniture_id in {i["furniture_id"] for i in scene}
    )
    category_by_item = {instance["furniture_id"]: instance["category"] for instance in instances}
    per_style = Counter(style for instance in instances for style in instance["styles"])
    per_style["(none)"] = sum(not instance["styles"] for instance in instances)
    repeated = sum(len(scene) != len({i["furniture_id"] for i in scene}) for scene in scenes.values())

    buckets = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10-49", 10, 49), ("50+", 50, math.inf)]
    per_bucket = {name: sum(low <= n <= high for n in scenes_per_item.values()) for name, low, high in buckets}

    pairs = _find_similar_pairs(scenes, 0.5, min_items)
    best_overlap = defaultdict(float)
    for overlap, scene_a, scene_b in pairs:
        best_overlap[scene_a] = max(best_overlap[scene_a], overlap)
        best_overlap[scene_b] = max(best_overlap[scene_b], overlap)
    near_identical = {t: sum(o >= t for o in best_overlap.values()) for t in (0.5, 0.7, 0.9)}

    stats = {
        "num_scenes": len(scenes),
        "num_instances": len(instances),
        "num_distinct_items": len(scenes_per_item),
        "scene_sizes": dict(sorted(scene_sizes.items())),
        "instances_per_category": dict(per_category.most_common()),
        "scenes_per_item_buckets": per_bucket,
        "instances_per_style": dict(per_style.most_common()),
        "num_scenes_with_repeated_item": repeated,
        "most_common_items": [
            (furniture_id, category_by_item[furniture_id], n) for furniture_id, n in scenes_per_item.most_common(10)
        ],
        "near_identical_scenes": near_identical,
    }

    sizes = list(scene_sizes.elements())
    print(f"Scenes: {len(scenes)}, item instances: {len(instances)}, distinct items: {len(scenes_per_item)}")
    print(
        f"Items per scene: median {sorted(sizes)[len(sizes) // 2]}, min {min(sizes)}, max {max(sizes)}; "
        f"{sum(n < 5 for n in sizes)} scenes have fewer than 5"
    )
    print(f"Scenes with the same item more than once: {repeated} ({repeated / len(scenes):.0%})")
    print(f"Items in only one scene: {per_bucket['1']} of {len(scenes_per_item)}")
    print(f"Item instances without a style: {per_style['(none)']} ({per_style['(none)'] / len(instances):.0%})")
    print("Most common items (scenes):", ", ".join(f"{c} {f} ({n})" for f, c, n in stats["most_common_items"][:5]))
    print(
        "Scenes sharing at least 50 / 70 / 90 % of their items with another scene: "
        + " / ".join(str(near_identical[t]) for t in (0.5, 0.7, 0.9))
    )

    # One series per chart, so one color; numbers live in the axes and the
    # printout above rather than on every bar.
    color, ink, grid = "#2a78d6", "#52514e", "#e6e5e1"

    def style_axes(ax, title, xlabel, ylabel, horizontal=False):
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylabel(ylabel, color=ink)
        ax.grid(axis="x" if horizontal else "y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ax = axes[0]
    ax.bar(list(scene_sizes), list(scene_sizes.values()), color=color, width=0.8)
    style_axes(ax, "Scene size", "Items in the scene", "Scenes")

    ax = axes[1]
    names = [name for name, _ in per_category.most_common()][::-1]
    ax.barh(names, [per_category[n] for n in names], color=color, height=0.7)
    style_axes(ax, "Items per category", "Item instances", "", horizontal=True)

    ax = axes[2]
    ax.bar(list(per_bucket), list(per_bucket.values()), color=color, width=0.7)
    style_axes(ax, "Item popularity (long tail)", "Number of scenes the item appears in", "Distinct items")

    ax = axes[3]
    names = [name for name, _ in per_style.most_common()][::-1]
    ax.barh(names, [per_style[n] for n in names], color=color, height=0.7)
    style_axes(ax, "Style labels", "Item instances", "", horizontal=True)

    ax = axes[4]
    labels = [f">= {int(t * 100)} %" for t in near_identical]
    ax.bar(labels, list(near_identical.values()), color=color, width=0.6)
    style_axes(ax, "Near-identical scenes", "Shares at least this share of its items with another scene", "Scenes")

    ax = axes[5]
    top = stats["most_common_items"][::-1]
    ax.barh([f"{c} {f}" for f, c, _ in top], [n for _, _, n in top], color=color, height=0.7)
    style_axes(ax, "Most common items", "Scenes the item appears in", "", horizontal=True)

    fig.suptitle(f"DeepFurniture: {len(scenes)} scenes, {len(scenes_per_item)} distinct items", fontsize=14)
    plt.tight_layout()
    plt.show()

    return stats


if __name__ == "__main__":
    visualize_random_scenes(num_scenes=3, sample_from_first_n=20)
    #find_and_show_duplicate_scenes(num_groups=3)
    #find_and_show_similar_scenes(num_pairs=3, min_overlap=0.4, max_overlap=0.45)
    #show_dataset_statistics()
