"""
What the model is asked for each scene: the prompt text and the JSON
schema of the answer.

The field order matters. The model writes the answer left to right, so it
first names what it sees ("reason") and only then picks the room type. For
a small model that short look-first step usually gives better labels than
answering with the label straight away.
"""

from src.data_generation.room_labelling import config


def build_schema():
    """
    JSON schema of the answer. Servers with structured output (vLLM, LM
    Studio) enforce it; with plain transformers it is only described in
    the prompt and checked afterwards (parsing.py).

    Returns:
        JSON schema dict.
    """
    properties = {
        "reason": {"type": "string"},
        "room_type": {"type": "string", "enum": list(config.ROOM_TYPES)},
        "confidence": {"type": "string", "enum": list(config.CONFIDENCE_LEVELS)},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_prompt():
    """
    The instruction text sent together with the scene image. It is the
    same for every scene.

    Returns:
        Prompt string.
    """
    room_lines = [f'- "{name}": {description}.' for name, description in config.ROOM_TYPES.items()]
    confidence = ", ".join(f'"{c}"' for c in config.CONFIDENCE_LEVELS)

    lines = [
        "The image is a rendered view of one room in an apartment, designed by an interior designer.",
        "Decide which type of room it is, judging by the furniture in view.",
        "",
        "Room types:",
        *room_lines,
        "",
        "Answer with a JSON object with exactly these fields, in this order:",
        '- "reason": one short sentence naming the furniture that decides the room type (at most 20 words).',
        '- "room_type": one of the room types above, written exactly as given.',
        f'- "confidence": one of {confidence}. Use "low" if only a corner of the room is visible '
        "or the room type is a guess.",
        "",
        "Answer with the JSON object only.",
    ]
    return "\n".join(lines)
