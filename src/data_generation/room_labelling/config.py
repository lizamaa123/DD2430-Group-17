"""
Settings shared by the room-labelling scripts: paths, the default model and
the room types the model chooses from.

Bump LABEL_VERSION whenever the room types or the prompt (prompts.py)
change, so answers from old and new instructions are never mixed up (the
version is part of the output file name).
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to this file, so scripts work from any working directory)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEEPFURNITURE_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "data")
# Original layout: scenes/<scene_id>/image.jpg (1600x1200).
SCENES_DIR = os.path.join(DEEPFURNITURE_DIR, "scenes")
EXCLUDED_SCENES_PATH = os.path.join(REPO_ROOT, "src", "data_generation", "deepfurniture_excluded_scenes.txt")

# Everything these scripts write goes here (data/ is git-ignored).
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "room_labels")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
HAND_LABELS_PATH = os.path.join(OUTPUT_DIR, "hand_labels.csv")

# Smaller copies of the scene images, made by shrink_scene_images.py. Flat
# layout: <scene_id>.jpg. Used instead of the originals when it exists
# (it is what gets uploaded to Kaggle).
SMALL_IMAGE_SIDE = 768
SMALL_IMAGES_DIR = os.path.join(OUTPUT_DIR, f"scene_images_{SMALL_IMAGE_SIDE}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Same model as the full captioning run (fits on Kaggle's T4 in float16).
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"

# Increase when ROOM_TYPES or the prompt change.
LABEL_VERSION = "v1"

# The answer is short: one sentence + two labels.
MAX_NEW_TOKENS = 120

# Greedy first; unusable answers are retried with sampling (Qwen's
# recommended non-thinking settings).
RETRY_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Room types
# ---------------------------------------------------------------------------

# Label -> the description shown to the model. Based on looking at samples
# of the scenes: mostly bedrooms, living rooms and hallways; some dining
# areas, studies and children's rooms; no kitchens or bathrooms (they are
# left out on purpose, those scenes would go to "other").
# The first five are the synthetic room types (room_generator.ANCHOR_RULES).
ROOM_TYPES = {
    "bedroom": "a bedroom for adults or teenagers: the main furniture is a bed (a desk may also be there)",
    "living_room": "a living room: sofa or armchairs, coffee table, TV wall; no dining table in view",
    "dining_room": "a dining room or dining area: a dining table with chairs is the main furniture",
    "office": "a home office or study: a desk with a chair, often bookshelves or a computer; no bed",
    "kids_room": "a room for young children: a small or themed bed, toys, play furniture or a playroom",
    "living_dining": "an open-plan room with both a living area (sofa) and a dining area (dining table) in view",
    "hallway": "a hallway, corridor or entrance area: mostly doors, walls, shoe or storage cabinets, "
               "a console table; no main room furniture",
    "other": "anything else, or a room that is (nearly) empty or can't be identified",
}

CONFIDENCE_LEVELS = ("high", "medium", "low")
