"""
Loading data for the pipeline: the DeepFurniture items to caption (one
per furniture image, with their category and style labels), the pilot
sample, the IKEA descriptions used for comparison, and the caption files
the pipeline writes.
"""

import json
import os
import random
import re
from collections import defaultdict

import pandas as pd

from src.data_generation.image_to_text import config


def load_deepfurniture_items(metadata_dir=config.METADATA_DIR, images_dir=config.FURNITURE_IMAGES_DIR):
    """
    Read every DeepFurniture item that has an image on disk.

    Items are captioned once per furniture_id, not once per scene
    instance: the same item can appear in many scenes (and several times
    in one scene), and its render is identical each time.

    Args:
        metadata_dir: Folder with furnitures.jsonl, categories.json and
            styles.json.
        images_dir: Folder with one <furniture_id>.jpg per item.

    Returns:
        List of dicts sorted by furniture_id, each with keys
        "furniture_id" (str), "df_category" (e.g. "cabinet#shelf"),
        "df_styles" (list of style names, possibly empty) and
        "image_path".
    """
    with open(os.path.join(metadata_dir, "categories.json")) as f:
        categories = json.load(f)
    with open(os.path.join(metadata_dir, "styles.json")) as f:
        styles = json.load(f)

    items = []
    num_missing_images = 0
    with open(os.path.join(metadata_dir, "furnitures.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            furniture_id = str(entry["furniture_id"])
            image_path = os.path.join(images_dir, f"{furniture_id}.jpg")
            if not os.path.exists(image_path):
                num_missing_images += 1
                continue

            # Some entries repeat a style ID; keep each one once, in order.
            style_ids = list(dict.fromkeys(entry.get("style_ids") or []))
            items.append({
                "furniture_id": furniture_id,
                "df_category": categories[str(entry["category_id"])],
                "df_styles": [styles[str(s)] for s in style_ids],
                "image_path": image_path,
            })

    if num_missing_images:
        print(f"Skipped {num_missing_images} items without an image in {images_dir}")

    return sorted(items, key=lambda item: int(item["furniture_id"]))


def sample_pilot_items(items, per_category=14, seed=42):
    """
    Pick a fixed, category-stratified sample for the pilot.

    The same seed always gives the same items, so different models and
    prompts are compared on exactly the same images.

    Args:
        items: Output of load_deepfurniture_items.
        per_category: Items per DeepFurniture category (11 categories, so
            14 gives 154 items).
        seed: Random seed.

    Returns:
        List of item dicts, grouped by category.
    """
    by_category = defaultdict(list)
    for item in items:
        by_category[item["df_category"]].append(item)

    rng = random.Random(seed)
    sample = []
    for category in sorted(by_category):
        pool = by_category[category]
        sample.extend(rng.sample(pool, min(per_category, len(pool))))
    return sample


# A comma-separated part that is only a measurement, e.g. "80x28x202 cm",
# "60 cm", "20 mm", or a value cut off in the IKEA data such as "120/" or
# "52x36x222-". Parts with letters ("5-seat", "outdoor") are kept.
_MEASUREMENT_PART = re.compile(r"^[\d\s.,x/\-]+(cm|mm|m)?$", re.IGNORECASE)


def strip_measurements(text):
    """
    Remove sizes from a short description:
    "Bookcase, 80x28x202 cm" -> "Bookcase",
    "Corner sofa, 4-seat" -> "Corner sofa, 4-seat" (unchanged).

    Args:
        text: A short description (IKEA's or the model's).

    Returns:
        The text without measurement parts.
    """
    parts = [part.strip() for part in str(text).split(",")]
    kept = [part for part in parts if part and not _MEASUREMENT_PART.match(part)]
    return ", ".join(kept)


def load_ikea_items(csv_path=config.IKEA_CSV_PATH):
    """
    Read the IKEA catalogue items, with an extra column holding the short
    description without sizes (for comparing phrase lengths and checking
    the prompt examples).

    Args:
        csv_path: Path to processed_ikea_data.csv.

    Returns:
        DataFrame with the CSV's columns plus "description_no_size".
    """
    df = pd.read_csv(csv_path)
    df["item_id"] = df["item_id"].astype(int)
    df["description_no_size"] = df["short_description"].map(strip_measurements)
    return df


def load_captions(path, valid_only=False):
    """
    Read a caption file written by run_captioning.py (one JSON record per
    line). If an item was captioned more than once (e.g. retried), the
    last record wins.

    Args:
        path: Path to the .jsonl file.
        valid_only: Drop records whose answer couldn't be used.

    Returns:
        Dict furniture_id (str) -> record dict. Empty if the file doesn't
        exist yet.
    """
    records = {}
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["furniture_id"]] = record
    if valid_only:
        records = {k: r for k, r in records.items() if r["valid"]}
    return records


def find_captions_file(name):
    """
    Full path of a caption file given just its name (as printed at the end
    of run_captioning.py), e.g. "pilot_Qwen3.5-4B_full_v1.jsonl". A full
    path is also accepted.

    Args:
        name: File name in config.CAPTIONS_DIR, or a full path.

    Returns:
        The full path.

    Raises:
        FileNotFoundError: listing the caption files that do exist.
    """
    path = name if os.path.isabs(name) else os.path.join(config.CAPTIONS_DIR, name)
    if os.path.exists(path):
        return path
    available = sorted(os.listdir(config.CAPTIONS_DIR)) if os.path.isdir(config.CAPTIONS_DIR) else []
    available = [f for f in available if f.endswith(".jsonl")]
    raise FileNotFoundError(
        f"No caption file {name!r} in {config.CAPTIONS_DIR}. "
        f"Available: {', '.join(available) if available else 'none yet (run run_captioning.py first)'}"
    )
