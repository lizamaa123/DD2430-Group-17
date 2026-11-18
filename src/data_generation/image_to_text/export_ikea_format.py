"""
Write the DeepFurniture items as a CSV in the same format as
data/processed_ikea_data.csv, so the two datasets can be compared or
loaded with the same code.

Columns (same order as the IKEA file, without width/height/depth):

    item_id            - DeepFurniture furniture_id
    name               - empty (DeepFurniture has no product series)
    category           - the IKEA category the model chose
    short_description  - built from the model's answer (see VARIANT)
    designer           - empty (DeepFurniture has no designer)

How to use: set CAPTIONS_FILE below to the file name printed by
run_captioning.py and run this file. Only needs pandas.
"""

import os
import sys

# Make the repo root importable, so this file can be run directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from src.data_generation.image_to_text import config  # noqa: E402
from src.data_generation.image_to_text.build_texts import (  # noqa: E402
    TEXT_VARIANTS,
    compose_description,
    run_name,
)
from src.data_generation.image_to_text.dataset import find_captions_file, load_captions  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# The file name printed at the end of run_captioning.py.
CAPTIONS_FILE = "full_Qwen3.5-2B_full_v2.jsonl"

# What goes in short_description (see build_texts.py). "short" (the item
# type only) is the one that matches the IKEA short_description.
VARIANT = "caption"

DROP_UNCERTAIN = False  # True: leave out items the model was unsure about

# ===========================================================================

IKEA_COLUMNS = ["item_id", "name", "category", "short_description", "designer"]


def build_ikea_format_table(captions_path, variant="short", drop_uncertain=False):
    """
    One row per valid item, with the processed_ikea_data.csv columns.

    Args:
        captions_path: Caption .jsonl from run_captioning.py.
        variant: Which text variant becomes short_description.
        drop_uncertain: Leave out items the model marked as uncertain.

    Returns:
        DataFrame with columns IKEA_COLUMNS, sorted by item_id.
    """
    records = load_captions(captions_path, valid_only=True)

    rows = []
    for furniture_id, record in records.items():
        fields = record["fields"]
        if drop_uncertain and fields.get("uncertain"):
            continue
        rows.append({
            "item_id": int(furniture_id),
            "name": None,
            "category": fields["ikea_category"],
            "short_description": compose_description(fields, variant),
            "designer": None,
        })

    table = pd.DataFrame(rows, columns=IKEA_COLUMNS)
    return table.sort_values("item_id").reset_index(drop=True)


def output_path(captions_path, variant):
    return os.path.join(
        config.OUTPUT_DIR, f"processed_deepfurniture_{variant}_{run_name(captions_path)}.csv"
    )


def main():
    """Run with the SETTINGS at the top of this file."""
    if VARIANT not in TEXT_VARIANTS:
        raise ValueError(f"Unknown text variant '{VARIANT}'; choose from {TEXT_VARIANTS}")

    captions_path = find_captions_file(CAPTIONS_FILE)
    table = build_ikea_format_table(captions_path, variant=VARIANT, drop_uncertain=DROP_UNCERTAIN)
    if table["short_description"].isna().any():
        print(f"{table['short_description'].isna().sum()} items have no '{VARIANT}' "
              f"description in this caption file (short prompt?)")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = output_path(captions_path, VARIANT)
    table.to_csv(path, index=False)
    print(f"Wrote {len(table)} items to {path}")


if __name__ == "__main__":
    main()
