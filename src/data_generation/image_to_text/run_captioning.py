"""
Describe DeepFurniture item images with a vision-language model and save
the answers to a .jsonl file (one record per item).

How to use: change the SETTINGS below and run this file. The name of the
file the answers were saved to is printed at the end; build_texts.py and
quality_checks.py need that name.

The run can be stopped at any time. Running the file again with the same
settings continues where it stopped: items already saved are skipped.
"""

import json
import os
import sys
import time

# Make the repo root importable, so this file can be run directly (e.g.
# with VS Code's Run button) and not only as a module.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm  # noqa: E402

from src.data_generation.image_to_text import config, prompts  # noqa: E402
from src.data_generation.image_to_text.backends import Request, make_backend  # noqa: E402
from src.data_generation.image_to_text.dataset import (  # noqa: E402
    load_captions, load_deepfurniture_items, sample_pilot_items,
)
from src.data_generation.image_to_text.parsing import parse_answer  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# "pilot": a fixed sample of PILOT_ITEMS_PER_CATEGORY items per category.
# "full": every DeepFurniture item.
MODE = "pilot"

# "full": ask for every field. "short": only category and type (used once in
# the pilot, to check that the many fields don't change the short answer).
PROMPT_VARIANT = "full"

# How the model is run (see backends.py):
# "dry-run": no model, placeholder answers - for checking that the code works.
# "transformers": load the model in this process (GPU, Mac or CPU).
# "server": send the images to a running vLLM / LM Studio server.
BACKEND = "dry-run"

MODEL = config.DEFAULT_MODEL  # e.g. "Qwen/Qwen3.5-0.8B" for a quick test on a Mac

PILOT_ITEMS_PER_CATEGORY = 14  # 11 categories -> 154 items
PILOT_SEED = 42  # same seed = same pilot items, so runs can be compared

LIMIT = None  # e.g. 5 to only do the first 5 items while testing
BATCH_SIZE = 4  # items sent to the model at once
RETRY_INVALID = False  # True: redo items whose saved answer was unusable

# Only for BACKEND = "transformers"
DEVICE = None  # "cuda", "mps" or "cpu"; None picks automatically

# Only for BACKEND = "server"
SERVER_URL = "http://localhost:8000/v1"  # vLLM default; LM Studio uses http://localhost:1234/v1
SERVER_API_KEY = "EMPTY"  # any text works for a local server
SERVER_WORKERS = 8  # requests in flight at once; keep BATCH_SIZE at least this big
SERVER_ENFORCE_SCHEMA = True  # set to False if the server rejects the answer format

# None: name the file after the settings, e.g. pilot_Qwen3.5-4B_full_v1.jsonl
OUTPUT_FILE = None

# ===========================================================================


def default_output_path(mode, model_name, variant, backend_name):
    """
    Caption file name that encodes how it was made, e.g.
    captions/pilot_Qwen3.5-4B_full_v1.jsonl.
    """
    model_short = "dry-run" if backend_name == "dry-run" else model_name.split("/")[-1]
    filename = f"{mode}_{model_short}_{variant}_{config.PROMPT_VERSION}.jsonl"
    return os.path.join(config.CAPTIONS_DIR, filename)


def caption_items(items, backend, output_path, variant="full", batch_size=4, model_name="",
                  backend_name="", retry_invalid=False):
    """
    Describe every item that isn't in the output file yet and append the
    records to it, one batch at a time (so nothing is lost if the run
    stops).

    An answer that can't be used is retried with sampling, up to
    config.MAX_ATTEMPTS attempts in total; the last attempt is saved
    either way, with valid=False if it still failed.

    Args:
        items: Item dicts from dataset.py.
        backend: Object from backends.make_backend.
        output_path: .jsonl file to append to.
        variant: Prompt variant, "full" or "short".
        batch_size: Items per call to the backend.
        model_name: Stored in each record.
        backend_name: Stored in each record.
        retry_invalid: Also redo items whose saved answer is invalid.

    Returns:
        Dict with counts and timing for the items processed in this run.
    """
    done = load_captions(output_path)
    todo = [
        item for item in items
        if item["furniture_id"] not in done or (retry_invalid and not done[item["furniture_id"]]["valid"])
    ]
    print(f"{len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to go -> {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats = {"processed": 0, "valid": 0, "seconds": 0.0}

    with open(output_path, "a") as out, tqdm(total=len(todo), unit="item") as progress:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            requests = [
                Request(
                    furniture_id=item["furniture_id"],
                    image_path=item["image_path"],
                    prompt=prompts.build_prompt(item["df_category"], variant),
                    schema=prompts.build_schema(item["df_category"], variant),
                )
                for item in batch
            ]

            batch_start = time.perf_counter()
            results = [None] * len(batch)
            pending = list(range(len(batch)))
            for attempt in range(1, config.MAX_ATTEMPTS + 1):
                sampling = None if attempt == 1 else config.RETRY_SAMPLING
                raws = backend.generate([requests[i] for i in pending], sampling=sampling)
                still_pending = []
                for i, raw in zip(pending, raws):
                    fields, problems, valid = parse_answer(raw, batch[i]["df_category"], variant)
                    results[i] = (raw, fields, problems, valid, attempt)
                    if not valid:
                        still_pending.append(i)
                pending = still_pending
                if not pending:
                    break
            seconds_per_item = (time.perf_counter() - batch_start) / len(batch)

            for item, (raw, fields, problems, valid, attempts) in zip(batch, results):
                record = {
                    "furniture_id": item["furniture_id"],
                    "df_category": item["df_category"],
                    "df_styles": item["df_styles"],
                    "model": model_name,
                    "backend": backend_name,
                    "prompt_variant": variant,
                    "prompt_version": config.PROMPT_VERSION,
                    "fields": fields,
                    "problems": problems,
                    "valid": valid,
                    "attempts": attempts,
                    "seconds": round(seconds_per_item, 3),
                    "raw": raw,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["valid"] += valid
            out.flush()

            stats["processed"] += len(batch)
            stats["seconds"] += seconds_per_item * len(batch)
            progress.update(len(batch))

    return stats


def main():
    """Run with the SETTINGS at the top of this file."""
    if MODE not in ("pilot", "full"):
        raise ValueError(f'MODE must be "pilot" or "full", not {MODE!r}')

    items = load_deepfurniture_items()
    total_items = len(items)
    if MODE == "pilot":
        items = sample_pilot_items(items, per_category=PILOT_ITEMS_PER_CATEGORY, seed=PILOT_SEED)
    if LIMIT:
        items = items[:LIMIT]

    backend = make_backend(
        BACKEND,
        model_name=MODEL,
        device=DEVICE,
        base_url=SERVER_URL,
        api_key=SERVER_API_KEY,
        workers=SERVER_WORKERS,
        use_schema=SERVER_ENFORCE_SCHEMA,
    )
    if BACKEND == "server" and BATCH_SIZE < SERVER_WORKERS:
        print(f"Note: BATCH_SIZE ({BATCH_SIZE}) is smaller than SERVER_WORKERS ({SERVER_WORKERS}); "
              f"raise BATCH_SIZE to keep all workers busy.")

    if OUTPUT_FILE:
        output_path = os.path.join(config.CAPTIONS_DIR, OUTPUT_FILE)
    else:
        output_path = default_output_path(MODE, MODEL, PROMPT_VARIANT, BACKEND)

    stats = caption_items(
        items, backend, output_path,
        variant=PROMPT_VARIANT,
        batch_size=BATCH_SIZE,
        model_name=MODEL if BACKEND != "dry-run" else "dry-run",
        backend_name=BACKEND,
        retry_invalid=RETRY_INVALID,
    )

    if stats["processed"]:
        per_item = stats["seconds"] / stats["processed"]
        print(f"Valid answers: {stats['valid']}/{stats['processed']}")
        print(f"Time per item: {per_item:.2f} s -> all {total_items} items would take "
              f"about {per_item * total_items / 3600:.1f} h with these settings")
    print(f"\nAnswers are in: {os.path.basename(output_path)}")
    print("Use this name as CAPTIONS_FILE in build_texts.py and quality_checks.py.")


if __name__ == "__main__":
    main()
