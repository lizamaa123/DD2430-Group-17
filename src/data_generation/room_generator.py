"""
Synthetic Room Generation Engine — starter skeleton

  columns: item_id, name, category, short_description, designer, width, height, depth
  - "name" is actually the IKEA series/product-line (EKET, BILLY, IVAR, ...),
    so we use it as the style anchor instead of a separate series column.
  - "category" values below follow the standard IKEA catalog taxonomy this
    dataset uses (Bar furniture, Beds, Bookcases & shelving units, ...).
    Run `df['category'].unique()` yourself and double check ANCHOR_RULES
    below actually matches your file before trusting the output — this is
    a best-guess mapping, not verified against your CSV.
"""

import random
import pandas as pd
import json

# 0. Load metadata
METADATA_PATH = "data/processed_ikea_data.csv"
df = pd.read_csv(METADATA_PATH)

# 1. Functional anchor rules
# For each room type: "required" categories must appear; "optional" ones
# are sampled in with some probability to add variety.
# CATEGORY STRINGS MUST MATCH df['category'] EXACTLY — verify and edit these.
ANCHOR_RULES = {
    "bedroom": {
        "required": ["Beds"],
        "optional": ["Wardrobes", "Chests of drawers & drawer units",
                     "Bookcases & shelving units"],
    },
    "living_room": {
        "required": ["Sofas & armchairs", "Tables & desks"],
        "optional": ["Bookcases & shelving units", "TV & media furniture",
                     "Room dividers"],
    },
    "dining_room": {
        "required": ["Tables & desks", "Chairs"],
        "optional": ["Sideboards, buffets & console tables",
                      "Cabinets & cupboards"],
    },
    "office": {
        "required": ["Tables & desks", "Chairs"],
        "optional": ["Bookcases & shelving units", "Cabinets & cupboards"],
    },
    "kids_room": {
        "required": ["Children's furniture"],
        "optional": ["Nursery furniture", "Chests of drawers & drawer units"],
    },
}
OPTIONAL_ITEM_PROB = 0.6  # chance each optional category gets included


# 2. Style-constrained sampling
def sample_item(category, df, style_key="name", style_value=None):
    """Sample one item of `category`, preferring one matching the style
    constraint (same series, via the 'name' column) if possible, else fall
    back to any item of that category."""
    pool = df[df["category"] == category]
    if pool.empty:
        return None

    if style_key and style_value is not None:
        styled = pool[pool[style_key] == style_value]
        if not styled.empty:
            pool = styled

    return pool.sample(1).iloc[0]


def generate_room(room_type, df, style_key="name"):
    """Build one synthetic room following anchor rules + style constraint."""
    rules = ANCHOR_RULES[room_type]
    items = []

    # Anchor the room's style on the first required item's series ("name").
    first_category = rules["required"][0]
    seed_item = sample_item(first_category, df, style_key)
    if seed_item is None:
        return None
    style_value = seed_item[style_key]
    items.append(seed_item)

    for category in rules["required"][1:]:
        item = sample_item(category, df, style_key, style_value)
        if item is not None:
            items.append(item)

    for category in rules["optional"]:
        if random.random() < OPTIONAL_ITEM_PROB:
            item = sample_item(category, df, style_key, style_value)
            if item is not None:
                items.append(item)

    return items


# 3. Generate the full dataset
def generate_dataset(n_rooms, df, room_types=None):
    room_types = room_types or list(ANCHOR_RULES.keys())
    rooms = []

    for room_id in range(n_rooms):
        room_type = random.choice(room_types)
        items = generate_room(room_type, df)
        if not items:
            continue
        rooms.append({
            "room_id": room_id,
            "room_type": room_type,
            # int(...) converts pandas' numpy.int64 to a plain Python int,
            # since json.dump() can't serialize numpy types directly.
            "item_ids": [int(it["item_id"]) for it in items],
        })

    return rooms


# ---------------------------------------------------------------------------
# 4. Room-level train/val/test split
# ---------------------------------------------------------------------------
def split_rooms(rooms, train=0.8, val=0.1, test=0.1, seed=42):
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


# 5. Run + save

if __name__ == "__main__":
    N_ROOMS = 50_000 
    rooms = generate_dataset(N_ROOMS, df)
    rooms = split_rooms(rooms)

    with open("data/synthetic_rooms.json", "w") as f:
        json.dump(rooms, f, indent=2)

    print(f"Generated {len(rooms)} rooms")
    print(f"train={sum(r['split']=='train' for r in rooms)}, "
          f"val={sum(r['split']=='val' for r in rooms)}, "
          f"test={sum(r['split']=='test' for r in rooms)}")
