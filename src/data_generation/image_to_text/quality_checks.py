"""
Checks for a caption run, mainly for the pilot.

How to use: set CAPTIONS_FILE below to the file name printed by
run_captioning.py and run this file. It always reports:
- validity: how many answers were usable, which problems occurred, how
  often the model was uncertain, time per item.
- categories: which IKEA category each DeepFurniture category was mapped to.
- styles: agreement between the model's style guesses and DeepFurniture's
  labels, for items that have labels.
- type length: words in "type" compared with IKEA descriptions (sizes
  removed).

Optional (see SETTINGS):
- COMPARE_FILE: whether the short prompt gives the same category/type as
  the full prompt on the same items.
- WRITE_GRADING_SHEET: a CSV for grading answers by hand next to the images.
- PROBE_VARIANT: needs build_texts.py to have been run on the same
  captions. Trains a classifier to tell DeepFurniture from IKEA
  embeddings (accuracy near 0.5 = similar, near 1.0 = easy to tell apart)
  and prints the nearest IKEA items for a few DeepFurniture items.
"""

import os
import random
import sys
from collections import Counter, defaultdict

# Make the repo root importable, so this file can be run directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_generation.image_to_text import config, prompts  # noqa: E402
from src.data_generation.image_to_text.dataset import (  # noqa: E402
    find_captions_file, load_captions, load_deepfurniture_items, load_ikea_items,
)


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# The file name printed at the end of run_captioning.py.
CAPTIONS_FILE = "pilot_Qwen3.5-0.8B_full_v1.jsonl"

# A second run on the same items to compare with (e.g. the short prompt),
# or None.
COMPARE_FILE = "pilot_Qwen3.5-4B_short_v1.jsonl"

# Write <captions file>_grading.csv: every answer with empty columns to
# mark right/wrong by hand while looking at the images.
WRITE_GRADING_SHEET = True

# "short", "medium", "rich" or "caption" to compare those embeddings with
# the IKEA embeddings; None to skip. Run build_texts.py first.
PROBE_VARIANT = None

# The probe only uses furniture by default (IKEA has no lamps, doors, ...).
PROBE_ALL_CATEGORIES = False

# ===========================================================================


def check_prompt_examples(ikea):
    """Print IKEA_EXAMPLES phrases that don't occur in the IKEA data (sizes removed)."""
    known = set(ikea["description_no_size"])
    missing = [
        (category, phrase)
        for category, phrases in config.IKEA_EXAMPLES.items()
        for phrase in phrases
        if phrase not in known
    ]
    if missing:
        print("IKEA_EXAMPLES phrases not found in the IKEA data:")
        for category, phrase in missing:
            print(f"  {category}: {phrase!r}")


def summarize(records, total_items):
    """Validity, problems, uncertainty and timing."""
    n = len(records)
    valid = [r for r in records if r["valid"]]
    print(f"\n== Validity ({n} items)")
    print(f"valid: {len(valid)}/{n}")
    print(f"needed a retry: {sum(r['attempts'] > 1 for r in records)}")

    problems = Counter(p.split(":")[0] for r in records for p in r["problems"])
    for problem, count in problems.most_common():
        print(f"  {problem}: {count}")

    by_category = defaultdict(list)
    for r in valid:
        if "uncertain" in r["fields"]:
            by_category[r["df_category"]].append(bool(r["fields"]["uncertain"]))
    if by_category:
        print("uncertain, per DeepFurniture category:")
        for category, flags in sorted(by_category.items()):
            print(f"  {category:15s} {sum(flags)}/{len(flags)}")

    seconds = [r["seconds"] for r in records]
    per_item = sum(seconds) / len(seconds)
    print(f"time per item: {per_item:.2f} s (all {total_items} items: about {per_item * total_items / 3600:.1f} h)")


def category_mapping(records):
    """Which IKEA categories each furniture category was mapped to."""
    print("\n== IKEA categories chosen")
    counts = defaultdict(Counter)
    for r in records:
        if r["valid"] and prompts.asks_for_ikea_category(r["df_category"]):
            counts[r["df_category"]][r["fields"]["ikea_category"]] += 1
    for category, counter in sorted(counts.items()):
        print(f"  {category}: " + ", ".join(f"{c} ({k})" for c, k in counter.most_common()))


def style_agreement(records):
    """Compare the model's styles with DeepFurniture's labels."""
    labelled = [r for r in records if r["valid"] and r["df_styles"] and "style" in r["fields"]]
    if not labelled:
        return
    any_overlap = sum(bool(set(r["df_styles"]) & set(r["fields"]["style"])) for r in labelled)
    exact = sum(set(r["df_styles"]) == set(r["fields"]["style"]) for r in labelled)
    print(f"\n== Styles ({len(labelled)} items with DeepFurniture labels)")
    print(f"at least one style in common: {any_overlap}/{len(labelled)}")
    print(f"same set of styles:           {exact}/{len(labelled)}")

    # For each DeepFurniture style, what the model said most often.
    said = defaultdict(Counter)
    for r in labelled:
        for label in r["df_styles"]:
            said[label].update(r["fields"]["style"])
    print("model's styles for each DeepFurniture label:")
    for label, counter in sorted(said.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {label:18s} -> " + ", ".join(f"{s} ({k})" for s, k in counter.most_common(3)))

    unlabelled = sum(1 for r in records if r["valid"] and not r["df_styles"])
    print(f"items without DeepFurniture style label (filled by the model): {unlabelled}")


def type_lengths(records, ikea):
    """Words in "type" vs IKEA descriptions without sizes."""
    ours = [len(r["fields"]["type"].split()) for r in records if r["valid"]]
    theirs = [len(d.split()) for d in ikea["description_no_size"] if d]
    print("\n== Length of the type phrase (words)")
    print(f"model: mean {np.mean(ours):.1f}, median {np.median(ours):.0f}, max {max(ours)}")
    print(f"IKEA:  mean {np.mean(theirs):.1f}, median {np.median(theirs):.0f}, max {max(theirs)}")


def compare_runs(records, other_records):
    """Category/type agreement between two runs on the same items."""
    other = {r["furniture_id"]: r for r in other_records}
    pairs = [(r, other[r["furniture_id"]]) for r in records
             if r["valid"] and r["furniture_id"] in other and other[r["furniture_id"]]["valid"]]
    if not pairs:
        print("\nNo items in common with the comparison run")
        return
    same_category = sum(a["fields"]["ikea_category"] == b["fields"]["ikea_category"] for a, b in pairs)
    same_type = sum(a["fields"]["type"].lower() == b["fields"]["type"].lower() for a, b in pairs)
    print(f"\n== Comparison with the other run ({len(pairs)} items in common)")
    print(f"same ikea_category: {same_category}/{len(pairs)}")
    print(f"same type:          {same_type}/{len(pairs)}")
    print("examples where type differs:")
    shown = 0
    for a, b in pairs:
        if a["fields"]["type"].lower() != b["fields"]["type"].lower():
            print(f"  {a['furniture_id']:>10s}  {a['fields']['type']!r:40s} vs {b['fields']['type']!r}")
            shown += 1
            if shown == 15:
                break


def write_grading_sheet(records, items_by_id, path):
    """CSV with the answers and empty columns to grade them by hand."""
    rows = []
    for r in records:
        fields = r["fields"]
        rows.append({
            "furniture_id": r["furniture_id"],
            "image_path": items_by_id[r["furniture_id"]]["image_path"] if r["furniture_id"] in items_by_id else "",
            "df_category": r["df_category"],
            "df_styles": "|".join(r["df_styles"]),
            "valid": r["valid"],
            "ikea_category": fields.get("ikea_category", ""),
            "type": fields.get("type", ""),
            "colour": fields.get("colour", ""),
            "material": fields.get("material", ""),
            "details": fields.get("details", ""),
            "style": "|".join(fields.get("style", [])),
            "caption": fields.get("caption", ""),
            "uncertain": fields.get("uncertain", ""),
            # Fill in by hand: 1 = correct, 0 = wrong, empty = not judged.
            "ok_category": "",
            "ok_type": "",
            "ok_colour": "",
            "ok_material": "",
            "ok_style": "",
            "notes": "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote grading sheet to {path}")


def domain_probe(df_embeddings, ikea_embeddings, seed=42):
    """
    5-fold cross-validated accuracy of a logistic regression telling
    DeepFurniture embeddings from IKEA embeddings, on equally many of each.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(seed)
    df_x = np.stack([v.numpy() for v in df_embeddings.values()])
    ikea_x = np.stack([v.numpy() for v in ikea_embeddings.values()])
    n = min(len(df_x), len(ikea_x))
    df_x = df_x[rng.choice(len(df_x), n, replace=False)]
    ikea_x = ikea_x[rng.choice(len(ikea_x), n, replace=False)]
    x = np.concatenate([df_x, ikea_x])
    y = np.array([0] * n + [1] * n)
    scores = cross_val_score(LogisticRegression(max_iter=2000), x, y, cv=5)
    return scores.mean(), n


def nearest_ikea(table, variant, df_embeddings, ikea_embeddings, ikea, k=3, num_items=10, seed=42):
    """Print the k most similar IKEA items for a few DeepFurniture items."""
    from src.data_generation.item_encoder import build_item_text

    ikea_ids = list(ikea_embeddings)
    ikea_matrix = np.stack([ikea_embeddings[i].numpy() for i in ikea_ids])
    ikea_text = {int(row["item_id"]): build_item_text(row) for _, row in ikea.iterrows()}

    rng = random.Random(seed)
    sample = table.dropna(subset=[f"text_{variant}"])
    sample = sample[sample["furniture_id"].isin(df_embeddings)]
    sample = sample.iloc[rng.sample(range(len(sample)), min(num_items, len(sample)))]

    print(f"\n== Nearest IKEA items ('{variant}' texts)")
    for _, row in sample.iterrows():
        similarities = ikea_matrix @ df_embeddings[row["furniture_id"]].numpy()
        print(f"\nDF {row['furniture_id']}: {row[f'text_{variant}']}")
        for index in np.argsort(-similarities)[:k]:
            print(f"   {similarities[index]:.2f}  {ikea_text.get(ikea_ids[index], ikea_ids[index])}")


def main():
    """Run with the SETTINGS at the top of this file."""
    captions_path = find_captions_file(CAPTIONS_FILE)

    ikea = load_ikea_items()
    items = load_deepfurniture_items()
    items_by_id = {item["furniture_id"]: item for item in items}
    records = list(load_captions(captions_path).values())

    check_prompt_examples(ikea)
    summarize(records, total_items=len(items))
    category_mapping(records)
    style_agreement(records)
    type_lengths(records, ikea)

    if COMPARE_FILE:
        try:
            compare_path = find_captions_file(COMPARE_FILE)
        except FileNotFoundError as error:
            print(f"\nSkipping the comparison: {error}")
        else:
            compare_runs(records, list(load_captions(compare_path).values()))

    if WRITE_GRADING_SHEET:
        sheet = os.path.splitext(captions_path)[0] + "_grading.csv"
        write_grading_sheet(records, items_by_id, sheet)

    if PROBE_VARIANT:
        import torch

        from src.data_generation.image_to_text.build_texts import embeddings_path, texts_path

        df_embeddings_path = embeddings_path(captions_path, PROBE_VARIANT)
        if not os.path.exists(df_embeddings_path):
            print(f"\nSkipping the domain probe: {df_embeddings_path} doesn't exist. "
                  f"Run build_texts.py with CAPTIONS_FILE = {CAPTIONS_FILE!r} first.")
            return

        df_embeddings = torch.load(df_embeddings_path, weights_only=False)
        ikea_embeddings = torch.load(config.IKEA_EMBEDDINGS_PATH, weights_only=False)
        table = pd.read_csv(texts_path(captions_path))

        if not PROBE_ALL_CATEGORIES:
            furniture = set(table.loc[table["df_category"].isin(config.DF_TO_IKEA_CATEGORIES), "furniture_id"])
            df_embeddings = {k: v for k, v in df_embeddings.items() if k in furniture}

        accuracy, n = domain_probe(df_embeddings, ikea_embeddings)
        print(f"\n== Domain probe ('{PROBE_VARIANT}' texts, {n} items per dataset)")
        print(f"accuracy telling DeepFurniture from IKEA: {accuracy:.3f} (0.5 = indistinguishable)")
        nearest_ikea(table, PROBE_VARIANT, df_embeddings, ikea_embeddings, ikea)


if __name__ == "__main__":
    main()
