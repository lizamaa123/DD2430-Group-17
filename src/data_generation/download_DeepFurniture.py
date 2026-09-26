import os
import tarfile

from huggingface_hub import hf_hub_download

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
    OBS: Takes up about 16 GB of memory and takes about 10 minutes to run with the default 
    settings. 
    
    Download one archive from data/<subfolder>/ in the DeepFurniture repo,
    extract only the members whose extension is in keep_extensions, then
    delete the downloaded archive.

    Args:
        subfolder: Dataset folder the archive lives in on the Hub
            ("furnitures" or "scenes").
        archive_name: Filename of the archive, e.g. "scenes_0007.tar.gz".
        keep_extensions: Tuple of lowercase file extensions to keep from
            inside the archive; every other member is skipped.
        cache_dir: Local root directory; extracted files are written to
            cache_dir/data/<subfolder>/.

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


def download_deepfurniture_subset(
    include_scenes=True,
    num_furniture_archives=NUM_FURNITURE_ARCHIVES,
    num_scene_archives=NUM_SCENE_ARCHIVES,
    cache_dir=DATA_DIR,
):
    """
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

    for i in range(num_furniture_archives):
        archive_name = f"furnitures_{i:04d}.tar.gz"
        _download_and_extract_archive("furnitures", archive_name, KEEP_EXTENSIONS["furnitures"], cache_dir)

    if include_scenes:
        for i in range(num_scene_archives):
            archive_name = f"scenes_{i:04d}.tar.gz"
            _download_and_extract_archive("scenes", archive_name, KEEP_EXTENSIONS["scenes"], cache_dir)


if __name__ == "__main__":
    download_deepfurniture_subset()
