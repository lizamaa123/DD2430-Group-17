"""
Turning the model's raw answer into a clean record, and checking it.

Small deviations (e.g. "living room" instead of "living_room", or
"Bedroom") are snapped to the allowed label and noted as a problem, so
they can be counted in the pilot. An answer without a usable room type is
invalid and gets retried.
"""

import difflib
import re

from src.data_generation.image_to_text.parsing import extract_json
from src.data_generation.room_labelling import config


def _normalise(value):
    """Lower case, spaces/hyphens -> underscores: "Living room" -> "living_room"."""
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def _snap(value, options):
    """Return the option matching value after normalising, the closest one, or None."""
    value = _normalise(value)
    if value in options:
        return value
    close = difflib.get_close_matches(value, list(options), n=1, cutoff=0.6)
    return close[0] if close else None


def parse_answer(raw):
    """
    Parse and check one answer.

    Args:
        raw: The model's raw answer.

    Returns:
        (fields, problems, valid):
        fields - dict with "reason", "room_type" and "confidence" (empty if
            the answer is unusable). "confidence" is None if missing.
        problems - list of short strings describing what was wrong or
            fixed, e.g. "room_type snapped". Empty if the answer was clean.
        valid - False if there is no JSON or no usable room type.
    """
    data = extract_json(raw)
    if data is None:
        return {}, ["no JSON object"], False

    problems = []
    given = str(data.get("room_type", "")).strip()
    room_type = _snap(given, config.ROOM_TYPES) if given else None
    if room_type is None:
        return {}, [f"room_type not allowed: {given!r}"], False
    if room_type != given:
        problems.append("room_type snapped")

    confidence = _snap(data.get("confidence", ""), config.CONFIDENCE_LEVELS) if data.get("confidence") else None
    if confidence is None:
        problems.append("missing confidence")
    elif confidence != data["confidence"]:
        problems.append("confidence snapped")

    reason = re.sub(r"\s+", " ", str(data.get("reason", ""))).strip()
    if not reason:
        problems.append("missing reason")

    return {"reason": reason, "room_type": room_type, "confidence": confidence}, problems, True
