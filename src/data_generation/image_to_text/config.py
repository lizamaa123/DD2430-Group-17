"""
Settings shared by the image-to-text pipeline: paths, the default model,
how DeepFurniture categories map onto the IKEA categories, the style
vocabulary, and the IKEA example phrases shown to the model.

Everything that is a judgement call (which IKEA categories a DeepFurniture
category may become, which examples are shown, ...) lives here so it can
be changed in one place. Bump PROMPT_VERSION whenever something here or in
prompts.py changes what the model is asked, so old and new outputs can be
told apart in the caption files.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to this file, so scripts work from any working directory)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEEPFURNITURE_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "data")
FURNITURE_IMAGES_DIR = os.path.join(DEEPFURNITURE_DIR, "furnitures")
METADATA_DIR = os.path.join(DEEPFURNITURE_DIR, "metadata")

IKEA_CSV_PATH = os.path.join(REPO_ROOT, "data", "processed_ikea_data.csv")
IKEA_EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "data", "item_embeddings.pt")

# Everything this pipeline writes goes here (data/ is git-ignored).
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "image_to_text")
CAPTIONS_DIR = os.path.join(OUTPUT_DIR, "captions")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Qwen3.5 small models (0.8B / 2B / 4B / 9B) are Apache-2.0 and take images
# natively. 4B is the starting point; compare against 9B in the pilot.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# Increase when the prompt, schema, examples or category mapping change.
# v2: stronger "describe only the item, not the background" instruction.
PROMPT_VERSION = "v2"

MAX_NEW_TOKENS = 300

# Greedy decoding is used first (deterministic). If the answer can't be
# parsed, it is retried with sampling, since greedy would only repeat the
# same answer. These are Qwen's recommended non-thinking settings.
RETRY_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

# The IKEA categories each DeepFurniture furniture category may be mapped to.
# The model only chooses within this list. IKEA's own taxonomy is messy
# (e.g. "Bedside table" is under Tables & desks, "Armchair" appears under
# both Chairs and Sofas & armchairs, sofa-beds are under Beds), which is why
# some lists include categories that look surprising at first.
DF_TO_IKEA_CATEGORIES = {
    "cabinet#shelf": [
        "Bookcases & shelving units",
        "Cabinets & cupboards",
        "Wardrobes",
        "Chests of drawers & drawer units",
        "TV & media furniture",
        "Sideboards, buffets & console tables",
        "Tables & desks",
    ],
    "table": [
        "Tables & desks",
        "Bar furniture",
        "Café furniture",
        "Outdoor furniture",
    ],
    "chair#stool": [
        "Chairs",
        "Sofas & armchairs",
        "Bar furniture",
        "Café furniture",
        "Outdoor furniture",
    ],
    "sofa": [
        "Sofas & armchairs",
        "Beds",
        "Outdoor furniture",
    ],
    "bed": [
        "Beds",
        "Children's furniture",
        "Nursery furniture",
    ],
}

# DeepFurniture categories with no IKEA counterpart in our data. The model
# is not asked for a category; this fixed name is used instead.
DF_FIXED_CATEGORY = {
    "lamp": "Lighting",
    "door": "Doors",
    "curtain": "Curtains & blinds",
    "plant": "Plants",
    "decoration": "Decoration",
    "home-appliance": "Home appliances",
}

# How DeepFurniture's category labels are shown to the model.
DF_CATEGORY_DISPLAY = {
    "cabinet#shelf": "cabinet or shelf",
    "chair#stool": "chair or stool",
    "home-appliance": "home appliance",
}

# ---------------------------------------------------------------------------
# Styles (DeepFurniture's own vocabulary, so the model's guesses can be
# compared with the existing labels)
# ---------------------------------------------------------------------------

# styles.json name -> how it is shown to the model. "electic" is a typo in
# the dataset; the model sees "eclectic" and the answer is mapped back.
STYLE_DISPLAY = {
    "modern": "modern",
    "country": "country",
    "European#American": "European/American",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Mediterranean": "Mediterranean",
    "Southeast-Asian": "Southeast Asian",
    "Nordic": "Nordic",
    "Industrial": "industrial",
    "electic": "eclectic",
    "other": "other",
}

MAX_STYLES = 3

# ---------------------------------------------------------------------------
# Example phrases for the "type" field
# ---------------------------------------------------------------------------

# Real IKEA short_descriptions from processed_ikea_data.csv, with the sizes
# removed (strip_measurements in dataset.py). They show the model the terse
# catalogue phrasing. Checked against the CSV by quality_checks.py.
IKEA_EXAMPLES = {
    "Bar furniture": ["Bar stool with backrest", "Bar table", "Bar stool"],
    "Beds": ["Bed frame", "Bed frame with storage", "Bunk bed frame", "Divan bed", "2-seat sofa-bed"],
    "Bookcases & shelving units": ["Shelving unit", "Bookcase", "Wall-mounted cabinet combination"],
    "Cabinets & cupboards": ["Glass-door cabinet", "Cabinet with doors", "Storage combination with doors"],
    "Café furniture": ["Chair with armrests", "Table", "Stool"],
    "Chairs": ["Chair", "Armchair", "Swivel chair", "Stool", "Chaise longue"],
    "Chests of drawers & drawer units": ["Chest of 3 drawers", "Chest of 6 drawers", "Drawer unit on castors"],
    "Children's furniture": ["Children's table", "Children's chair", "Desk with shelf unit"],
    "Nursery furniture": ["Changing table", "Changing table with drawers"],
    "Outdoor furniture": ["Table, outdoor", "Armchair, outdoor", "3-seat modular sofa, outdoor"],
    "Sideboards, buffets & console tables": ["Console table", "Storage combination with doors"],
    "Sofas & armchairs": ["3-seat sofa", "2-seat sofa", "Corner sofa, 4-seat", "Footstool", "Armchair"],
    "TV & media furniture": ["TV bench", "TV bench with doors", "TV storage combination"],
    "Tables & desks": ["Bedside table", "Coffee table", "Desk", "Side table", "Dressing table"],
    "Wardrobes": ["Wardrobe with 2 doors", "Open wardrobe", "Corner wardrobe"],
}

# For the categories IKEA doesn't sell in our data there are no real
# examples, so these are written by us in the same style.
NON_IKEA_EXAMPLES = {
    "lamp": ["Floor lamp", "Table lamp", "Pendant lamp", "Wall lamp"],
    "door": ["Door", "Double door", "Sliding door"],
    "curtain": ["Curtains, 1 pair", "Roller blind"],
    "plant": ["Potted plant", "Plant in pot"],
    "decoration": ["Framed picture", "Vase", "Set of 3 framed pictures", "Mirror"],
    "home-appliance": ["Fridge-freezer", "Washing machine", "Television"],
}

MAX_EXAMPLES_IN_PROMPT = 8
