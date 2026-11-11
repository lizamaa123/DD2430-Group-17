"""
Show item pictures next to the model's answers, to check by eye that the
descriptions are right.

How to use: set CAPTIONS_FILE below to the file name printed by
run_captioning.py and run this file. The items are shown a page at a time;
close the window to see the next page. Set SAVE_PAGES = True to save the
pages as images instead.
"""

import os
import random
import sys
import textwrap

# Make the repo root importable, so this file can be run directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.data_generation.image_to_text import config  # noqa: E402
from src.data_generation.image_to_text.dataset import find_captions_file, load_captions  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# The file name printed at the end of run_captioning.py.
CAPTIONS_FILE = "pilot_Qwen3.5-0.8B_full_v2.jsonl"

ITEMS_PER_PAGE = 6  # shown as 2 rows of 3
MAX_PAGES = 5  # stop after this many pages; None for all

# Only show some items (None = no filter):
CATEGORY = None  # a DeepFurniture category, e.g. "lamp" or "cabinet#shelf"
ONLY = None  # "uncertain", "problems" (anything flagged by the checks) or "invalid"

SHUFFLE = False  # True: random order (same order every time for the same SEED)
SEED = 0

# True: save each page as a .png in data/deepfurniture/image_to_text/previews/
# instead of opening a window.
SAVE_PAGES = False

# ===========================================================================

COLUMNS = 3
TEXT_WIDTH = 42  # characters per line under each picture


def select_records(records):
    """Apply the CATEGORY / ONLY / SHUFFLE settings."""
    selected = list(records.values())
    if CATEGORY:
        selected = [r for r in selected if r["df_category"] == CATEGORY]
    if ONLY == "uncertain":
        selected = [r for r in selected if r["fields"].get("uncertain")]
    elif ONLY == "problems":
        selected = [r for r in selected if r["problems"]]
    elif ONLY == "invalid":
        selected = [r for r in selected if not r["valid"]]
    elif ONLY is not None:
        raise ValueError(f'ONLY must be None, "uncertain", "problems" or "invalid", not {ONLY!r}')
    if SHUFFLE:
        random.Random(SEED).shuffle(selected)
    return selected


def describe(record):
    """The text shown under one picture."""
    fields = record["fields"]
    df_styles = ", ".join(record["df_styles"]) or "none"

    if not record["valid"]:
        lines = ["UNUSABLE ANSWER: " + "; ".join(record["problems"]), "", "Raw answer:", record["raw"][:300]]
    else:
        lines = [
            f"IKEA category: {fields['ikea_category']}",
            f"Type: {fields['type']}",
        ]
        if "colour" in fields:  # not there for the short prompt
            lines += [
                f"Colour: {fields['colour']}",
                f"Material: {fields['material']}",
                f"Details: {fields['details']}",
                f"Style: {', '.join(fields['style']) or '-'}   (DeepFurniture: {df_styles})",
                f"Caption: {fields['caption']}",
            ]
        if fields.get("uncertain"):
            lines.append("MODEL IS UNCERTAIN")
        if record["problems"]:
            lines.append("Flags: " + "; ".join(record["problems"]))

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, TEXT_WIDTH, subsequent_indent="   ") or [""])
    return "\n".join(wrapped)


def show_page(records, page_number, total_pages):
    """Draw one page of items: picture on top, answer underneath."""
    rows = (len(records) + COLUMNS - 1) // COLUMNS
    fig = plt.figure(figsize=(COLUMNS * 4.6, rows * 7.0))
    # Each row of items gets two grid rows: the pictures, then the answers.
    grid = fig.add_gridspec(rows * 2, COLUMNS, height_ratios=[1.0, 0.62] * rows,
                            left=0.02, right=0.98, top=0.93, bottom=0.01, wspace=0.12, hspace=0.12)

    for index, record in enumerate(records):
        row, column = divmod(index, COLUMNS)
        image_ax = fig.add_subplot(grid[2 * row, column])
        text_ax = fig.add_subplot(grid[2 * row + 1, column])
        image_ax.axis("off")
        text_ax.axis("off")

        image_path = os.path.join(config.FURNITURE_IMAGES_DIR, f"{record['furniture_id']}.jpg")
        if os.path.exists(image_path):
            image_ax.imshow(Image.open(image_path).convert("RGB"))
        image_ax.set_title(f"{record['furniture_id']}  ({record['df_category']})", fontsize=10)

        # Red text = something to look at: unusable, uncertain or flagged.
        flagged = not record["valid"] or record["fields"].get("uncertain") or record["problems"]
        text_ax.text(0.0, 1.0, describe(record), transform=text_ax.transAxes, va="top", ha="left",
                     fontsize=8.5, family="monospace", color="darkred" if flagged else "black")

    fig.suptitle(f"{CAPTIONS_FILE}  -  page {page_number}/{total_pages}", fontsize=12)

    if SAVE_PAGES:
        folder = os.path.join(config.OUTPUT_DIR, "previews")
        os.makedirs(folder, exist_ok=True)
        name = os.path.splitext(CAPTIONS_FILE)[0]
        path = os.path.join(folder, f"{name}_page{page_number:03d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Saved {path}")
    else:
        plt.show()


def main():
    """Run with the SETTINGS at the top of this file."""
    records = select_records(load_captions(find_captions_file(CAPTIONS_FILE)))
    if not records:
        print("No items match the settings.")
        return

    pages = [records[i:i + ITEMS_PER_PAGE] for i in range(0, len(records), ITEMS_PER_PAGE)]
    if MAX_PAGES:
        pages = pages[:MAX_PAGES]
    print(f"Showing {sum(len(p) for p in pages)} of {len(records)} matching items on {len(pages)} pages")

    for number, page in enumerate(pages, start=1):
        show_page(page, number, len(pages))


if __name__ == "__main__":
    main()
