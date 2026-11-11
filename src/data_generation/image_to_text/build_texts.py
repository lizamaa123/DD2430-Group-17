"""
Turn the model's answers into item texts in the same format as the IKEA
items, and embed them with the same sentence encoder.

The IKEA side is not changed. Each DeepFurniture item is written as a row
with the IKEA columns (name, category, short_description, designer) and
passed through the unchanged item_encoder.build_item_text, e.g.

    Product series: . Category: Tables & desks. Description: Bedside table with 3 drawers. Designer: .

Series and designer don't exist for DeepFurniture, so they stay empty.
Only short_description differs between the text variants:

    short   - type                                   "Bedside table with 3 drawers"
    medium  - type, colour                           "..., white body, brown top"
    rich    - type, colour, material, details        "..., painted wood, panelled fronts"
    caption - the model's one-sentence caption

Only "short" carries the same kind of information as the IKEA texts, so
it is the variant for training on one dataset and testing on the other.

How to use: set CAPTIONS_FILE below to the file name printed by
run_captioning.py and run this file. It writes a CSV with all the texts
(to read through) and one embeddings file per text variant.
"""

import os
import sys

# Make the repo root importable, so this file can be run directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_generation.image_to_text import config  # noqa: E402
from src.data_generation.image_to_text.dataset import find_captions_file, load_captions  # noqa: E402

TEXT_VARIANTS = ("short", "medium", "rich", "caption")


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# The file name printed at the end of run_captioning.py.
CAPTIONS_FILE = "pilot_Qwen3.5-4B_full_v2.jsonl"

# Which text versions to build (see the top of this file).
VARIANTS = ["short", "medium", "rich", "caption"]

DROP_UNCERTAIN = False  # True: leave out items the model was unsure about
MAKE_EMBEDDINGS = True  # False: only write the texts CSV

# ===========================================================================

# Which answer fields go into short_description for each variant.
_VARIANT_FIELDS = {
    "short": ("type",),
    "medium": ("type", "colour"),
    "rich": ("type", "colour", "material", "details"),
    "caption": ("caption",),
}


def compose_description(fields, variant):
    """
    The short_description of one item for one text variant.

    Args:
        fields: The "fields" dict of a caption record.
        variant: One of TEXT_VARIANTS.

    Returns:
        The description, or None if the record lacks the needed fields
        (e.g. "rich" from a run with the short prompt).
    """
    if any(name not in fields for name in _VARIANT_FIELDS[variant]):
        return None
    if variant == "caption":
        return fields["caption"].rstrip(".") or None
    parts = [fields[name] for name in _VARIANT_FIELDS[variant] if fields[name]]
    return ", ".join(parts)


def build_text_table(captions_path, variants=TEXT_VARIANTS, drop_uncertain=False):
    """
    One row per valid item, with its labels and the full item text for
    every variant.

    Args:
        captions_path: Caption .jsonl from run_captioning.py.
        variants: Text variants to build.
        drop_uncertain: Leave out items the model marked as uncertain.

    Returns:
        DataFrame with columns furniture_id, df_category, ikea_category,
        uncertain, df_styles, vlm_styles and text_<variant> per variant.
    """
    # Imported here so the rest of the package works without
    # sentence-transformers installed (item_encoder imports it).
    from src.data_generation.item_encoder import build_item_text

    all_records = load_captions(captions_path)
    records = {k: r for k, r in all_records.items() if r["valid"]}
    if len(records) < len(all_records):
        print(f"Skipping {len(all_records) - len(records)} items with invalid answers "
              f"(run run_captioning.py with RETRY_INVALID = True to redo them)")

    rows = []
    for furniture_id, record in records.items():
        fields = record["fields"]
        if drop_uncertain and fields.get("uncertain"):
            continue
        row = {
            "furniture_id": int(furniture_id),
            "df_category": record["df_category"],
            "ikea_category": fields["ikea_category"],
            "uncertain": fields.get("uncertain"),
            "df_styles": "|".join(record["df_styles"]),
            "vlm_styles": "|".join(fields.get("style", [])),
        }
        for variant in variants:
            description = compose_description(fields, variant)
            if description is None:
                continue
            # Same columns as processed_ikea_data.csv, so build_item_text
            # treats it exactly like an IKEA item.
            ikea_style_row = {
                "name": np.nan,
                "category": fields["ikea_category"],
                "short_description": description,
                "designer": np.nan,
            }
            row[f"text_{variant}"] = build_item_text(ikea_style_row)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("furniture_id").reset_index(drop=True)


def embed_texts(table, variant, model_name=None, batch_size=128):
    """
    Embed one text variant with the same encoder and settings as
    item_encoder.create_embeddings.

    Args:
        table: Output of build_text_table.
        variant: Which text_<variant> column to embed.
        model_name: Sentence encoder; defaults to item_encoder.MODEL_NAME.
        batch_size: Encoding batch size.

    Returns:
        Dict furniture_id (int) -> 384-d tensor, the same layout as
        data/item_embeddings.pt. Kept in a separate file: one
        DeepFurniture furniture_id equals an IKEA item_id.
    """
    from sentence_transformers import SentenceTransformer

    from src.data_generation.item_encoder import MODEL_NAME

    column = f"text_{variant}"
    subset = table.dropna(subset=[column])
    model = SentenceTransformer(model_name or MODEL_NAME)
    embeddings = model.encode(
        subset[column].tolist(),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {int(fid): emb.cpu() for fid, emb in zip(subset["furniture_id"], embeddings)}


def run_name(captions_path):
    """Caption file name without extension, used to name the outputs."""
    return os.path.splitext(os.path.basename(captions_path))[0]


def texts_path(captions_path):
    return os.path.join(config.OUTPUT_DIR, f"item_texts_{run_name(captions_path)}.csv")


def embeddings_path(captions_path, variant):
    return os.path.join(config.OUTPUT_DIR, f"item_embeddings_{variant}_{run_name(captions_path)}.pt")


def main():
    """Run with the SETTINGS at the top of this file."""
    unknown = [v for v in VARIANTS if v not in TEXT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown text variants {unknown}; choose from {TEXT_VARIANTS}")

    captions_path = find_captions_file(CAPTIONS_FILE)
    table = build_text_table(captions_path, variants=VARIANTS, drop_uncertain=DROP_UNCERTAIN)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    table.to_csv(texts_path(captions_path), index=False)
    print(f"Wrote {len(table)} items to {texts_path(captions_path)}")

    if not MAKE_EMBEDDINGS:
        return

    import torch

    for variant in VARIANTS:
        if f"text_{variant}" not in table or table[f"text_{variant}"].isna().all():
            print(f"No '{variant}' texts in this caption file (short prompt?); skipping")
            continue
        embeddings = embed_texts(table, variant)
        torch.save(embeddings, embeddings_path(captions_path, variant))
        print(f"Saved {len(embeddings)} '{variant}' embeddings to {embeddings_path(captions_path, variant)}")


if __name__ == "__main__":
    main()
