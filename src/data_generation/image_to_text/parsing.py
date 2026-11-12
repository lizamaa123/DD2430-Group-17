"""
Turning the model's raw answer into a clean record, and checking it.

With the server backend the schema is enforced, so most checks never
trigger. With plain transformers the model can still write invalid JSON or
a category that isn't allowed, which is caught here. Small deviations
(e.g. "Chests of drawers" instead of "Chests of drawers & drawer units")
are snapped to the closest allowed value and noted as a problem, so they
can be counted in the pilot.

Apart from that, nothing in the answer is changed or removed: possible
mistakes (sizes in "type", mentions of the background, ...) are only
flagged, so they can be looked at in show_captions.py / quality_checks.py
before deciding whether to remove them automatically.
"""

import difflib
import json
import re

from src.data_generation.image_to_text import config, prompts
from src.data_generation.image_to_text.dataset import strip_measurements

# Words that shouldn't be in "type" (they belong in colour/material). Only
# used to flag answers for review, not to reject them. "glass" and "light"
# are left out on purpose: "Glass-door cabinet" and "Pendant light" are
# fine types.
_COLOUR_MATERIAL_WORDS = {
    "white", "black", "grey", "gray", "brown", "beige", "blue", "green", "red", "yellow",
    "pink", "orange", "purple", "gold", "golden", "silver", "cream", "dark",
    "wooden", "wood", "oak", "walnut", "pine", "birch", "metal", "leather",
    "fabric", "velvet", "marble", "rattan", "plastic",
}

_STYLE_NAME_BY_DISPLAY = {display.lower(): name for name, display in config.STYLE_DISPLAY.items()}

# Every render has the same grey wall and floor behind the item, so words
# about it probably describe the photo setup, not the item ("on a gray
# background", "on a plain wall"). They are only flagged, not removed: a
# decoration's picture can have a background of its own, and a wall lamp is
# mounted on a wall.
_BACKGROUND_MENTION = re.compile(
    r"\b(?:background|backdrop|render(?:ed|ing)?|studio|shadows?|"
    r"(?:plain|grey|gray|blank|white|neutral)\s+(?:wall|floor|surface))\b",
    re.IGNORECASE,
)


def mentions_background(text):
    """True if the text seems to talk about the image setup rather than the item."""
    return bool(_BACKGROUND_MENTION.search(text))


def extract_json(raw):
    """
    Find the JSON object in a raw answer. Tolerates code fences, a leftover
    <think> block or text around the object.

    Args:
        raw: The model's answer.

    Returns:
        The parsed dict, or None if no JSON object could be read.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _snap(value, options):
    """Return the option matching value (case-insensitive), the closest one, or None."""
    lowered = {option.lower(): option for option in options}
    if value.lower() in lowered:
        return lowered[value.lower()]
    close = difflib.get_close_matches(value.lower(), list(lowered), n=1, cutoff=0.6)
    return lowered[close[0]] if close else None


def _clean_text(value):
    """Collapse whitespace and strip trailing punctuation from a string field."""
    return re.sub(r"\s+", " ", str(value)).strip().strip(".")


def parse_answer(raw, df_category, variant="full"):
    """
    Parse and check one answer.

    Args:
        raw: The model's raw answer.
        df_category: DeepFurniture category of the item.
        variant: Prompt variant the answer was generated with.

    Returns:
        (fields, problems, valid):
        fields - dict with the cleaned fields (empty if nothing could be
            parsed). Always has "ikea_category" when valid: the model's
            choice, or the fixed name for non-furniture categories.
        problems - list of short strings describing what was wrong or
            fixed, e.g. "ikea_category snapped". Empty if the answer was
            clean.
        valid - False if the answer can't be used (no JSON, no type, or no
            usable category); such items are retried.
    """
    problems = []
    data = extract_json(raw)
    if data is None:
        return {}, ["no JSON object"], False

    fields = {}

    # Category: chosen by the model for furniture, fixed otherwise.
    if prompts.asks_for_ikea_category(df_category):
        given = _clean_text(data.get("ikea_category", ""))
        options = prompts.allowed_ikea_categories(df_category)
        category = _snap(given, options) if given else None
        if category is None:
            return {}, [f"ikea_category not allowed: {given!r}"], False
        if category.lower() != given.lower():
            problems.append("ikea_category snapped")
        fields["ikea_category"] = category
    else:
        fields["ikea_category"] = config.DF_FIXED_CATEGORY[df_category]

    # Type: required, sentence case. Sizes are flagged, not removed.
    item_type = _clean_text(data.get("type", ""))
    if not item_type:
        return {}, ["empty type"], False
    if strip_measurements(item_type) != item_type:
        problems.append("type has measurements")
    item_type = item_type[0].upper() + item_type[1:]
    if set(re.findall(r"[a-z]+", item_type.lower())) & _COLOUR_MATERIAL_WORDS:
        problems.append("type mentions colour/material")
    fields["type"] = item_type

    if variant == "short":
        return fields, problems, True

    for name in ("colour", "material", "details", "caption"):
        value = data.get(name)
        if value is None:
            problems.append(f"missing {name}")
            value = ""
        text = _clean_text(value)
        if mentions_background(text):
            problems.append(f"{name} mentions background")
        fields[name] = text
    # Captions are sentences; keep their full stop.
    if fields["caption"]:
        fields["caption"] += "."

    # Styles: map the displayed names back to DeepFurniture's names.
    styles = data.get("style", [])
    if isinstance(styles, str):
        styles = [styles]
    mapped = []
    for style in styles if isinstance(styles, list) else []:
        name = _STYLE_NAME_BY_DISPLAY.get(str(style).strip().lower())
        if name is None:
            display = _snap(str(style).strip(), list(config.STYLE_DISPLAY.values()))
            name = _STYLE_NAME_BY_DISPLAY.get(display.lower()) if display else None
            problems.append("style snapped" if name else f"unknown style {style!r}")
        if name and name not in mapped:
            mapped.append(name)
    if not mapped:
        problems.append("no style")
    if len(mapped) > config.MAX_STYLES:
        problems.append("too many styles")
    fields["style"] = mapped[: config.MAX_STYLES]

    uncertain = data.get("uncertain")
    if isinstance(uncertain, str):
        uncertain = uncertain.strip().lower() == "true"
    if not isinstance(uncertain, bool):
        problems.append("missing uncertain")
        uncertain = None
    fields["uncertain"] = uncertain

    return fields, problems, True
