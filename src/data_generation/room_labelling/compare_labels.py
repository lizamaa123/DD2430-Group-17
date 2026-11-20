"""
Compare the model's room labels with the hand labels: accuracy, per room
type precision/recall, a confusion matrix, accuracy per confidence level,
and (optionally) a window showing the scenes the model got wrong.

How to use: set LABELS_FILE to the name printed by run_room_labelling.py,
label the pilot scenes with label_by_hand.py, then run this file.
"""

import os
import sys
import textwrap
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from src.data_generation.room_labelling import config  # noqa: E402
from src.data_generation.room_labelling.dataset import (  # noqa: E402
    find_labels_file, load_hand_labels, load_labels, load_scenes,
)


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

LABELS_FILE = "full_Qwen3.5-2B_v1.jsonl"  # name in data/deepfurniture/room_labels/labels/
HAND_LABELS_PATH = config.HAND_LABELS_PATH

SHOW_MISTAKES = True  # open a window with the scenes where model and hand label differ
MAX_MISTAKES_SHOWN = 20

# ===========================================================================


def compare(model_labels, hand_labels):
    """
    Put the model's and the hand labels side by side.

    Args:
        model_labels: Dict scene_id -> record from load_labels.
        hand_labels: Dict scene_id -> room type from load_hand_labels.

    Returns:
        DataFrame with one row per scene that has both labels: scene_id,
        hand, model, confidence, reason, correct. A scene whose model
        answer was invalid gets model = "(invalid)".
    """
    rows = []
    for scene_id, hand in hand_labels.items():
        record = model_labels.get(scene_id)
        if record is None:
            continue
        model = record["room_type"] if record["valid"] else "(invalid)"
        rows.append({"scene_id": scene_id, "hand": hand, "model": model,
                     "confidence": record.get("confidence"), "reason": record.get("reason") or "",
                     "correct": hand == model})
    return pd.DataFrame(rows)


def print_report(df):
    """Print accuracy, per-room-type scores, the confusion matrix and accuracy per confidence."""
    print(f"Scenes with both labels: {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.1%}\n")

    order = [t for t in config.ROOM_TYPES if t in set(df["hand"]) | set(df["model"])]
    order += sorted(set(df["model"]) - set(order))  # e.g. "(invalid)"

    print("Per room type (hand label = truth):")
    print(f"  {'room type':<14} {'hand':>5} {'model':>6} {'precision':>10} {'recall':>7}")
    for room_type in order:
        is_hand, is_model = df["hand"] == room_type, df["model"] == room_type
        both = (is_hand & is_model).sum()
        precision = f"{both / is_model.sum():.0%}" if is_model.sum() else "-"
        recall = f"{both / is_hand.sum():.0%}" if is_hand.sum() else "-"
        print(f"  {room_type:<14} {is_hand.sum():>5} {is_model.sum():>6} {precision:>10} {recall:>7}")

    print("\nConfusion matrix (rows: hand label, columns: model label):")
    matrix = pd.crosstab(df["hand"], df["model"]).reindex(index=order, columns=order, fill_value=0)
    matrix = matrix.loc[(matrix.sum(axis=1) > 0), (matrix.sum(axis=0) > 0)]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(matrix)

    print("\nAccuracy per confidence level:")
    for level, group in df.groupby(df["confidence"].fillna("(none)")):
        print(f"  {level:<8} {group['correct'].mean():>6.1%}  ({len(group)} scenes)")


def show_mistakes(df, max_shown=20):
    """Show the scenes where the labels differ, with both labels and the model's reason."""
    import matplotlib.pyplot as plt
    from PIL import Image

    mistakes = df[~df["correct"]].head(max_shown)
    if mistakes.empty:
        print("No mistakes to show.")
        return
    image_paths = {s["scene_id"]: s["image_path"] for s in load_scenes()}

    columns = 4
    rows = (len(mistakes) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (_, row) in zip(axes.flat, mistakes.iterrows()):
        axis.imshow(Image.open(image_paths[row["scene_id"]]))
        title = f"hand: {row['hand']} | model: {row['model']} ({row['confidence']})\n"
        title += "\n".join(textwrap.wrap(row["reason"], 55)[:2])
        axis.set_title(title, fontsize=8)
    figure.tight_layout()
    plt.show()


def main():
    """Run with the SETTINGS at the top of this file."""
    model_labels = load_labels(find_labels_file(LABELS_FILE))
    hand_labels = load_hand_labels(HAND_LABELS_PATH)
    if not hand_labels:
        raise FileNotFoundError(f"No hand labels in {HAND_LABELS_PATH}; run label_by_hand.py first.")

    problems = Counter(p for r in model_labels.values() for p in r["problems"])
    print(f"Model answers: {len(model_labels)}, valid: {sum(r['valid'] for r in model_labels.values())}")
    if problems:
        print("Problems: " + ", ".join(f"{p}: {n}" for p, n in problems.most_common()))
    print()

    df = compare(model_labels, hand_labels)
    if df.empty:
        print("No scene has both a model and a hand label. Are both made on the same pilot scenes?")
        return
    print_report(df)
    if SHOW_MISTAKES:
        show_mistakes(df, MAX_MISTAKES_SHOWN)


if __name__ == "__main__":
    main()
