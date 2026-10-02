"""
Synthetic Room Generation Engine

columns:
item_id, name, category, short_description, designer, width, height, depth

- "name" is the IKEA series/product-line (EKET, BILLY, IVAR, ...)
- We use it as a style anchor.
- Every generated room must contain at least MIN_ITEMS_PER_ROOM items.
"""

import random
import pandas as pd
import json


# 0. Load metadata
METADATA_PATH = "data/processed_ikea_data.csv"
df = pd.read_csv(METADATA_PATH)


# Configuration
N_ROOMS = 50_000
MIN_ITEMS_PER_ROOM = 3
OPTIONAL_ITEM_PROB = 0.6


# 1. Functional anchor rules
ANCHOR_RULES = {

    "bedroom": {
        "required": ["Beds"],
        "optional": [
            "Wardrobes",
            "Chests of drawers & drawer units",
            "Bookcases & shelving units"
        ],
    },

    "living_room": {
        "required": [
            "Sofas & armchairs",
            "Tables & desks"
        ],
        "optional": [
            "Bookcases & shelving units",
            "TV & media furniture",
            "Room dividers"
        ],
    },

    "dining_room": {
        "required": [
            "Tables & desks",
            "Chairs"
        ],
        "optional": [
            "Sideboards, buffets & console tables",
            "Cabinets & cupboards"
        ],
    },

    "office": {
        "required": [
            "Tables & desks",
            "Chairs"
        ],
        "optional": [
            "Bookcases & shelving units",
            "Cabinets & cupboards"
        ],
    },

    "kids_room": {
        "required": ["Children's furniture"],
        "optional": [
            "Nursery furniture",
            "Chests of drawers & drawer units"
        ],
    },
}


# 2. Item sampling
def sample_item(category, df):
    """Sample one random IKEA item from the given category."""

    # Take the dataframe and keep only rows whose categ match therequested cate
    pool = df[df["category"] == category]

    if pool.empty:
        return None
    # randomly choose one row and extract that
    return pool.sample(1).iloc[0]

# 3. Generate one room
def generate_room(room_type, df, min_items=MIN_ITEMS_PER_ROOM):

    """Build one synthetic room following the anchor rules."""

    rules = ANCHOR_RULES[room_type]
    items = []

    # First required item
    first_category = rules["required"][0]

    seed_item = sample_item(first_category, df)

    if seed_item is None:
        return None

    items.append(seed_item)

    # Remaining required categories
    for category in rules["required"][1:]:

        item = sample_item(category, df)

        if item is not None:
            items.append(item)

    # Optional categories
    unused_optional = []

    for category in rules["optional"]:

        if random.random() < OPTIONAL_ITEM_PROB:

            item = sample_item(category, df)

            if item is not None:
                items.append(item)

        else:
            unused_optional.append(category)

    # Make sure room contains at least MIN_ITEMS_PER_ROOM
    random.shuffle(unused_optional)

    for category in unused_optional:

        if len(items) >= min_items:
            break

        item = sample_item(category, df)

        if item is not None:
            items.append(item)

    # Discard room if it is still too small
    if len(items) < min_items:
        return None

    return items

# 4. Generate full dataset
def generate_dataset(n_rooms,df, room_types=None, min_items=MIN_ITEMS_PER_ROOM):

    room_types = room_types or list(ANCHOR_RULES.keys())

    rooms = []

    for room_id in range(n_rooms):

        room_type = random.choice(room_types)

        items = generate_room(room_type, df, min_items=min_items)

        if not items:
            continue

        rooms.append({
            "room_id": room_id,
            "room_type": room_type,
            "item_ids": [
                int(item["item_id"])
                for item in items
            ],
        })

    return rooms


# 5. Room-level train / validation / test split
def split_rooms(
    rooms,
    train=0.8,
    val=0.1,
    test=0.1,
    seed=42
):

    assert abs(train + val + test - 1.0) < 1e-6

    rng = random.Random(seed)

    shuffled = rooms[:]

    rng.shuffle(shuffled)

    n = len(shuffled)

    n_train = int(n * train)
    n_val = int(n * val)

    for i, room in enumerate(shuffled):

        if i < n_train:

            room["split"] = "train"

        elif i < n_train + n_val:

            room["split"] = "val"

        else:

            room["split"] = "test"

    return shuffled


# 6. Run + save
if __name__ == "__main__":

    rooms = generate_dataset(
        N_ROOMS,
        df,
        min_items=MIN_ITEMS_PER_ROOM
    )

    rooms = split_rooms(rooms)

    with open(
        "data/synthetic_rooms.json",
        "w"
    ) as f:

        json.dump(
            rooms,
            f,
            indent=2
        )


    print(f"Generated {len(rooms)} rooms")

    print(
        f"train={sum(r['split'] == 'train' for r in rooms)}, "
        f"val={sum(r['split'] == 'val' for r in rooms)}, "
        f"test={sum(r['split'] == 'test' for r in rooms)}"
    )

    sizes = [
        len(room["item_ids"])
        for room in rooms
    ]

    print(
        f"Room sizes: "
        f"min={min(sizes)}, "
        f"max={max(sizes)}, "
        f"mean={sum(sizes) / len(sizes):.2f}"
    )