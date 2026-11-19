"""
Make smaller copies of the scene images, for uploading to Kaggle and for
faster labelling.

The originals are 1600x1200 (about 13 GB for all scenes). The copies are
at most SMALL_IMAGE_SIDE pixels on the long side (about 2 GB), which is
plenty to see the room type and also means fewer image tokens for the
model. Excluded (duplicate) scenes are skipped. The copies go to
data/deepfurniture/room_labels/scene_images_<side>/<scene_id>.jpg, and
optionally into a .zip next to that folder, ready to upload as a Kaggle
dataset.

How to use: run this file. It can be stopped and run again; images that
already exist are skipped.
"""

import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data_generation.room_labelling import config  # noqa: E402
from src.data_generation.room_labelling.dataset import load_scenes  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# The long side of the copies. Change config.SMALL_IMAGE_SIDE instead to also
# change the folder the other scripts look in.
MAX_SIDE = config.SMALL_IMAGE_SIDE
JPEG_QUALITY = 90
OUTPUT_DIR = config.SMALL_IMAGES_DIR

# True: also write <OUTPUT_DIR>.zip for uploading to Kaggle.
MAKE_ZIP = True

# ===========================================================================


def shrink_image(source_path, target_path, max_side=768, quality=90):
    """
    Save a copy of an image with its long side at most max_side pixels.

    Args:
        source_path: The original image.
        target_path: Where to save the copy.
        max_side: Long side of the copy in pixels.
        quality: JPEG quality (1-95).
    """
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        image.save(target_path, "JPEG", quality=quality)


def folder_size_gb(folder):
    """Total size of the files in a folder, in GB."""
    return sum(entry.stat().st_size for entry in os.scandir(folder) if entry.is_file()) / 1e9


def main():
    """Run with the SETTINGS at the top of this file."""
    scenes = load_scenes(config.SCENES_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    todo = [s for s in scenes if not os.path.exists(os.path.join(OUTPUT_DIR, s["scene_id"] + ".jpg"))]
    print(f"{len(scenes)} scenes, {len(scenes) - len(todo)} already done, {len(todo)} to go -> {OUTPUT_DIR}")

    for scene in tqdm(todo, unit="image"):
        target = os.path.join(OUTPUT_DIR, scene["scene_id"] + ".jpg")
        # Write to a temporary name first, so a stopped run never leaves a
        # half-written image that would be skipped next time.
        shrink_image(scene["image_path"], target + ".tmp", MAX_SIDE, JPEG_QUALITY)
        os.replace(target + ".tmp", target)

    print(f"Images: {len(os.listdir(OUTPUT_DIR))}, {folder_size_gb(OUTPUT_DIR):.2f} GB")

    if MAKE_ZIP:
        print("Zipping ...")
        zip_path = shutil.make_archive(OUTPUT_DIR, "zip", root_dir=os.path.dirname(OUTPUT_DIR),
                                       base_dir=os.path.basename(OUTPUT_DIR))
        print(f"Zip: {zip_path} ({os.path.getsize(zip_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
