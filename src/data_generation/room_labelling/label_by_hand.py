"""
Label scenes by hand, to check the model's labels against (compare_labels.py).

Shows one scene image at a time; press a key to label it:
    1-8  the room types, in the order shown in the window title
    s    skip (can't tell / broken image)
    b    go back one scene and label it again
    q    quit (everything labelled so far is saved)

By default it shows the same scenes as the pilot in run_room_labelling.py
(same number and seed), so every hand label can be compared. Label without
looking at the model's answers first. Each key press is saved straight away
to data/deepfurniture/room_labels/hand_labels.csv; running the file again
continues with the scenes that have no hand label yet.

How to use: run this file (it opens a matplotlib window).
"""

import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.data_generation.room_labelling import config  # noqa: E402
from src.data_generation.room_labelling.dataset import load_scenes, sample_pilot_scenes  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# Keep these equal to PILOT_SCENES / PILOT_SEED in run_room_labelling.py.
NUM_SCENES = 150
SEED = 42

OUTPUT_PATH = config.HAND_LABELS_PATH

# ===========================================================================

ROOM_TYPES = list(config.ROOM_TYPES)
KEYS = {str(i + 1): room_type for i, room_type in enumerate(ROOM_TYPES)}
KEYS["s"] = "skip"


def read_done(path):
    """Scene ids that already have a hand label (or were skipped)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["scene_id"] for row in csv.DictReader(f)}


def append_label(path, scene_id, room_type):
    """Add one row to the CSV (creating it with a header if needed). Later rows win."""
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["scene_id", "room_type"])
        writer.writerow([scene_id, room_type])


def main():
    """Run with the SETTINGS at the top of this file."""
    sample = sample_pilot_scenes(load_scenes(), n=NUM_SCENES, seed=SEED)
    done = read_done(OUTPUT_PATH)
    todo = [s for s in sample if s["scene_id"] not in done]
    print(f"{len(sample)} scenes, {len(sample) - len(todo)} already labelled, {len(todo)} to go -> {OUTPUT_PATH}")
    if not todo:
        return

    legend = "   ".join(f"{key}: {room_type}" for key, room_type in KEYS.items())
    legend += "   b: back   q: quit"
    state = {"index": 0}
    # matplotlib uses "s" (save) and "q" (quit) itself; turn those off.
    for name in ("keymap.save", "keymap.quit"):
        plt.rcParams[name] = []
    figure, axes = plt.subplots(figsize=(12, 9.5))

    def show():
        scene = todo[state["index"]]
        axes.clear()
        axes.imshow(Image.open(scene["image_path"]))
        axes.set_axis_off()
        axes.set_title(f"{state['index'] + 1}/{len(todo)}   {scene['scene_id']}\n{legend}", fontsize=9)
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(figure)
        elif event.key == "b" and state["index"] > 0:
            state["index"] -= 1
            show()
        elif event.key in KEYS:
            scene = todo[state["index"]]
            append_label(OUTPUT_PATH, scene["scene_id"], KEYS[event.key])
            print(f"{scene['scene_id']}: {KEYS[event.key]}")
            state["index"] += 1
            if state["index"] == len(todo):
                print("All scenes labelled.")
                plt.close(figure)
            else:
                show()

    figure.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


if __name__ == "__main__":
    main()
