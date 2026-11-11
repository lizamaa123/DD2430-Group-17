"""
What the model is asked for each item: the prompt text and the JSON
schema of the answer.

Two prompt variants exist:
- "full": every field (category, type, colour, material, details, style,
  caption, uncertain). This is the one used for the real run.
- "short": only category and type. Used in the pilot to check that asking
  for the extra fields doesn't change the short description.

The field order matters. The model writes the answer left to right, so the
fields needed for the short text variant (ikea_category, type) come first
and can't be influenced by what it writes later.
"""

from src.data_generation.image_to_text import config

PROMPT_VARIANTS = ("full", "short")


def asks_for_ikea_category(df_category):
    """True if the model chooses the IKEA category for this DeepFurniture category."""
    return df_category in config.DF_TO_IKEA_CATEGORIES


def allowed_ikea_categories(df_category):
    """The IKEA categories the model may choose for this DeepFurniture category."""
    return config.DF_TO_IKEA_CATEGORIES.get(df_category, [])


def example_phrases(df_category):
    """
    The example "type" phrases shown for one item: real IKEA descriptions
    from the categories the item may be mapped to, or our own examples for
    categories IKEA doesn't sell in our data.

    Args:
        df_category: DeepFurniture category, e.g. "table".

    Returns:
        List of at most config.MAX_EXAMPLES_IN_PROMPT phrases, without
        duplicates.
    """
    if not asks_for_ikea_category(df_category):
        return config.NON_IKEA_EXAMPLES[df_category][: config.MAX_EXAMPLES_IN_PROMPT]

    # Take examples round-robin across the allowed categories, so every
    # category is represented even when the limit is reached.
    per_category = [config.IKEA_EXAMPLES.get(c, []) for c in allowed_ikea_categories(df_category)]
    phrases = []
    for i in range(max(len(p) for p in per_category)):
        for category_examples in per_category:
            if i < len(category_examples) and category_examples[i] not in phrases:
                phrases.append(category_examples[i])
    return phrases[: config.MAX_EXAMPLES_IN_PROMPT]


def build_schema(df_category, variant="full"):
    """
    JSON schema of the answer for one item. Servers that support
    structured output (e.g. vLLM, LM Studio) use it to make invalid
    answers impossible; with plain transformers it is only described in
    the prompt and the answer is checked afterwards (parsing.py).

    Args:
        df_category: DeepFurniture category of the item.
        variant: "full" or "short".

    Returns:
        JSON schema dict.
    """
    properties = {}
    if asks_for_ikea_category(df_category):
        properties["ikea_category"] = {"type": "string", "enum": allowed_ikea_categories(df_category)}
    properties["type"] = {"type": "string"}

    if variant == "full":
        properties.update({
            "colour": {"type": "string"},
            "material": {"type": "string"},
            "details": {"type": "string"},
            "style": {
                "type": "array",
                "items": {"type": "string", "enum": list(config.STYLE_DISPLAY.values())},
                "minItems": 1,
                "maxItems": config.MAX_STYLES,
            },
            "caption": {"type": "string"},
            "uncertain": {"type": "boolean"},
        })

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_prompt(df_category, variant="full"):
    """
    The instruction text sent together with the item's image.

    Args:
        df_category: DeepFurniture category of the item.
        variant: "full" or "short".

    Returns:
        Prompt string.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}; use one of {PROMPT_VARIANTS}")

    shown_category = config.DF_CATEGORY_DISPLAY.get(df_category, df_category)
    examples = "; ".join(f'"{p}"' for p in example_phrases(df_category))

    lines = [
        "The image shows a single item of furniture or home decor. Describe only the item "
        "itself: the plain grey wall, floor and lighting are the same in every image and are "
        "not part of the item.",
        f"The dataset labels it as: {shown_category}.",
        "",
        "Describe it as a JSON object with exactly these fields, in this order:",
    ]

    if asks_for_ikea_category(df_category):
        options = "; ".join(f'"{c}"' for c in allowed_ikea_categories(df_category))
        lines.append(f'- "ikea_category": the best-fitting IKEA catalogue category, one of: {options}.')

    lines.append(
        '- "type": what the item is, in the short style of an IKEA catalogue description '
        f"(2-6 words, sentence case). Examples of the style: {examples}. "
        "Do not mention colour, material, style or size here."
    )

    if variant == "full":
        styles = ", ".join(f'"{s}"' for s in config.STYLE_DISPLAY.values())
        lines += [
            '- "colour": the main colour or colours (at most 5 words).',
            '- "material": the visible material or materials (at most 6 words).',
            '- "details": up to 3 distinguishing design details, comma-separated.',
            f'- "style": a list of 1-{config.MAX_STYLES} styles that fit the item, chosen from: {styles}.',
            '- "caption": one sentence describing the item (at most 25 words).',
            '- "uncertain": true if you cannot identify the item confidently, the image shows '
            "several separate items, the item is cut off, or it does not look like a "
            f"{shown_category}; otherwise false.",
        ]

    lines += [
        "",
        "In every field: never mention the background, wall, floor, the scene's lighting or shadows, or "
        "that the image is a render or photo; never mention sizes, measurements, brands, logos "
        "or product names. Answer with the JSON object only.",
    ]
    return "\n".join(lines)
