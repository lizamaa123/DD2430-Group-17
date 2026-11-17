"""
Turn the DeepFurniture scenes into rooms in the same format as
data/synthetic_rooms.json, so the same RoomDataset and models can be used
on both datasets:

    {"room_id": "DVD3HFDYEJI4OKYQBT3WKSY8",
     "room_type": null,
     "item_ids": [2880315, 2639579, 2823756, 3465459],
     "split": "train"}

- room_id:   the DeepFurniture scene id (a string, unlike the synthetic ints).
- room_type: None for now; DeepFurniture has no room types (to be labelled later).
- item_ids:  the scene's furniture ids as ints, the same keys as the embedding
             files from image_to_text/build_texts.py. Repeated identical items
             are kept, and all categories (doors, lamps, ...) are kept.
- split:     random train/val/test split, done the same way as
             room_generator.split_rooms (same proportions and seed).

Scenes listed in deepfurniture_excluded_scenes.txt (duplicates) are left out.
The text version (short, caption, ...) is not part of this file: it is chosen
by which embeddings file is loaded together with it.

How to use: run this file. The settings are below.
"""

import json
import os
import random
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

SCENE_ITEMS_PATH = os.path.join(REPO_ROOT, "data", "deepfurniture", "data", "metadata", "scene_items.json")
EXCLUDED_SCENES_PATH = os.path.join(REPO_ROOT, "src", "data_generation", "deepfurniture_excluded_scenes.txt")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "deepfurniture", "deepfurniture_rooms.json")

# Optional: a caption file from run_captioning.py, to check that every item in
# the rooms has a description. Set to None to skip the check.
CAPTIONS_PATH = os.path.join(
    REPO_ROOT, "data", "deepfurniture", "image_to_text", "captions", "full_Qwen3.5-2B_full_v2.jsonl"
)

# Same as room_generator.split_rooms.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SEED = 42

# ===========================================================================


def load_excluded_scenes(path):
    """
    Scene ids to leave out, read the same way as download_DeepFurniture.py.

    Args:
        path: The excluded-scenes .txt file (one scene id per line, # = comment).

    Returns:
        Set of scene ids.
    """
    with open(path) as f:
        lines = (line.strip() for line in f)
        return {line for line in lines if line and not line.startswith("#")}


def build_rooms(scene_items_path, excluded_scenes_path):
    """
    One room per kept scene, without a split yet.

    Args:
        scene_items_path: DeepFurniture metadata/scene_items.json.
        excluded_scenes_path: The excluded-scenes .txt file.

    Returns:
        List of room dicts, sorted by room_id.
    """
    with open(scene_items_path) as f:
        scene_items = json.load(f)
    excluded = load_excluded_scenes(excluded_scenes_path)

    rooms = []
    for scene_id in sorted(scene_items):
        if scene_id in excluded:
            continue
        rooms.append({
            "room_id": scene_id,
            "room_type": None,
            "item_ids": [int(item["furniture_id"]) for item in scene_items[scene_id]],
        })

    print(f"Scenes in scene_items.json: {len(scene_items)}, excluded: {len(excluded)}, "
          f"kept: {len(rooms)}")
    return rooms


def split_rooms(rooms, train=TRAIN_FRACTION, val=VAL_FRACTION, test=TEST_FRACTION, seed=SEED):
    """
    Random room-level split, the same method as room_generator.split_rooms
    (copied, since importing room_generator loads the IKEA data).

    Args:
        rooms: List of room dicts.
        train, val, test: Fractions, summing to 1.
        seed: Random seed.

    Returns:
        The rooms in shuffled order, each with a "split" key.
    """
    assert abs(train + val + test - 1.0) < 1e-6

    rng = random.Random(seed)
    shuffled = rooms[:]
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * train)
    n_val = int(len(shuffled) * val)
    for i, room in enumerate(shuffled):
        if i < n_train:
            room["split"] = "train"
        elif i < n_train + n_val:
            room["split"] = "val"
        else:
            room["split"] = "test"
    return shuffled


def print_summary(rooms, captions_path=None):
    """Rooms per split, room sizes, and (optionally) items without a caption."""
    splits = Counter(room["split"] for room in rooms)
    print("Rooms per split: " + ", ".join(f"{s}={splits[s]}" for s in ("train", "val", "test")))

    sizes = sorted(len(room["item_ids"]) for room in rooms)
    print(f"Items per room: min={sizes[0]}, median={sizes[len(sizes) // 2]}, max={sizes[-1]}, "
          f"total={sum(sizes)}")
    print(f"Rooms with 1 item: {sizes.count(1)}, with 2 items: {sizes.count(2)}")

    repeated = sum(len(set(room["item_ids"])) < len(room["item_ids"]) for room in rooms)
    print(f"Rooms with a repeated identical item: {repeated}")

    if captions_path is None:
        return
    if not os.path.exists(captions_path):
        print(f"Caption file not found, skipping the caption check: {captions_path}")
        return
    captioned = set()
    with open(captions_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("valid"):
                captioned.add(int(record["furniture_id"]))
    room_items = {item_id for room in rooms for item_id in room["item_ids"]}
    missing = room_items - captioned
    print(f"Unique items in rooms: {len(room_items)}, without a valid caption: {len(missing)}")
    if missing:
        print(f"  e.g. {sorted(missing)[:10]}")


def main():
    """Run with the SETTINGS at the top of this file."""
    rooms = build_rooms(SCENE_ITEMS_PATH, EXCLUDED_SCENES_PATH)
    rooms = split_rooms(rooms)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rooms, f, indent=2)
    print(f"Wrote {len(rooms)} rooms to {OUTPUT_PATH}")

    print_summary(rooms, CAPTIONS_PATH)


if __name__ == "__main__":
    main()
