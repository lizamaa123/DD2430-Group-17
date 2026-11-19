"""
Label DeepFurniture scenes with a room type using a vision-language model
and save the answers to a .jsonl file (one record per scene).

How to use: change the SETTINGS below and run this file. The name of the
file the answers were saved to is printed at the end; compare_labels.py
needs that name.

On Kaggle: set IMAGES_DIR to the uploaded scene images and OUTPUT_DIR to
/kaggle/working (see README.md in this folder).

The run can be stopped at any time. Running the file again with the same
settings continues where it stopped: scenes already saved are skipped.
"""

import json
import os
import sys
import time
from collections import Counter

# Make the repo root importable, so this file can be run directly (e.g.
# with VS Code's Run button) and not only as a module.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm  # noqa: E402

from src.data_generation.image_to_text.backends import Request, make_backend  # noqa: E402
from src.data_generation.room_labelling import config, prompts  # noqa: E402
from src.data_generation.room_labelling.dataset import (  # noqa: E402
    load_labels, load_scenes, sample_pilot_scenes,
)
from src.data_generation.room_labelling.parsing import parse_answer  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

# "pilot": a fixed random sample of PILOT_SCENES scenes. "full": every scene.
MODE = "pilot"

# How the model is run (see image_to_text/backends.py):
# "dry-run": no model, placeholder answers - for checking that the code works.
# "transformers": load the model in this process (GPU, Mac or CPU).
# "server": send the images to a running vLLM / LM Studio server.
BACKEND = "dry-run"

MODEL = config.DEFAULT_MODEL  # e.g. "Qwen/Qwen3.5-0.8B" for a quick test on a Mac

PILOT_SCENES = 150
PILOT_SEED = 42  # same seed = same scenes, also the ones label_by_hand.py shows

LIMIT = None  # e.g. 5 to only do the first 5 scenes while testing, None for no limit
BATCH_SIZE = 4  # scenes sent to the model at once
RETRY_INVALID = False  # True: redo scenes whose saved answer was unusable

# None: the small images from shrink_scene_images.py if they exist, else the
# originals. On Kaggle e.g. "/kaggle/input/deepfurniture-scenes/scene_images_768".
IMAGES_DIR = None

# None: data/deepfurniture/room_labels/labels. On Kaggle "/kaggle/working".
OUTPUT_DIR = None

# None: name the file after the settings, e.g. pilot_Qwen3.5-2B_v1.jsonl
OUTPUT_FILE = None

# Only for BACKEND = "transformers"
DEVICE = None  # "cuda", "mps" or "cpu"; None picks automatically
# "float16" is about 10x faster on GPUs without bf16 support, e.g. Kaggle's T4.
DTYPE = "auto"

# Only for BACKEND = "server"
SERVER_URL = "http://localhost:8000/v1"  # vLLM default; LM Studio uses http://localhost:1234/v1
SERVER_API_KEY = "EMPTY"
SERVER_WORKERS = 8  # requests in flight at once; keep BATCH_SIZE at least this big
SERVER_ENFORCE_SCHEMA = True  # set to False if the server rejects the answer format

# ===========================================================================


def default_output_name(mode, model_name, backend_name):
    """Label file name that encodes how it was made, e.g. pilot_Qwen3.5-2B_v1.jsonl."""
    model_short = "dry-run" if backend_name == "dry-run" else model_name.split("/")[-1]
    return f"{mode}_{model_short}_{config.LABEL_VERSION}.jsonl"


def label_scenes(scenes, backend, output_path, batch_size=4, model_name="", backend_name="",
                 retry_invalid=False):
    """
    Label every scene that isn't in the output file yet and append the
    records to it, one batch at a time (so nothing is lost if the run
    stops).

    An answer that can't be used is retried with sampling, up to
    config.MAX_ATTEMPTS attempts in total; the last attempt is saved
    either way, with valid=False if it still failed.

    Args:
        scenes: Scene dicts from dataset.load_scenes.
        backend: Object from backends.make_backend.
        output_path: .jsonl file to append to.
        batch_size: Scenes per call to the backend.
        model_name: Stored in each record.
        backend_name: Stored in each record.
        retry_invalid: Also redo scenes whose saved answer is invalid.

    Returns:
        Dict with counts and timing for the scenes processed in this run.
    """
    done = load_labels(output_path)
    todo = [
        scene for scene in scenes
        if scene["scene_id"] not in done or (retry_invalid and not done[scene["scene_id"]]["valid"])
    ]
    print(f"{len(scenes)} scenes, {len(scenes) - len(todo)} already done, {len(todo)} to go -> {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompt, schema = prompts.build_prompt(), prompts.build_schema()
    stats = {"processed": 0, "valid": 0, "seconds": 0.0}

    with open(output_path, "a") as out, tqdm(total=len(todo), unit="scene") as progress:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            requests = [Request(item_id=s["scene_id"], image_path=s["image_path"], prompt=prompt, schema=schema)
                        for s in batch]

            batch_start = time.perf_counter()
            results = [None] * len(batch)
            pending = list(range(len(batch)))
            for attempt in range(1, config.MAX_ATTEMPTS + 1):
                sampling = None if attempt == 1 else config.RETRY_SAMPLING
                raws = backend.generate([requests[i] for i in pending], sampling=sampling)
                still_pending = []
                for i, raw in zip(pending, raws):
                    fields, problems, valid = parse_answer(raw)
                    results[i] = (raw, fields, problems, valid, attempt)
                    if not valid:
                        still_pending.append(i)
                pending = still_pending
                if not pending:
                    break
            seconds_per_scene = (time.perf_counter() - batch_start) / len(batch)

            for scene, (raw, fields, problems, valid, attempts) in zip(batch, results):
                record = {
                    "scene_id": scene["scene_id"],
                    "room_type": fields.get("room_type"),
                    "confidence": fields.get("confidence"),
                    "reason": fields.get("reason"),
                    "model": model_name,
                    "backend": backend_name,
                    "label_version": config.LABEL_VERSION,
                    "problems": problems,
                    "valid": valid,
                    "attempts": attempts,
                    "seconds": round(seconds_per_scene, 3),
                    "raw": raw,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["valid"] += valid
            out.flush()

            stats["processed"] += len(batch)
            stats["seconds"] += seconds_per_scene * len(batch)
            progress.update(len(batch))

    return stats


def print_distribution(records):
    """Print how many scenes got each room type and confidence."""
    valid = [r for r in records.values() if r["valid"]]
    if not valid:
        return
    print(f"\nRoom types ({len(valid)} valid answers):")
    for room_type, count in Counter(r["room_type"] for r in valid).most_common():
        print(f"  {room_type:<14} {count:>6}  ({count / len(valid):.1%})")
    confidence = Counter(r["confidence"] for r in valid)
    print("Confidence: " + ", ".join(f"{c}: {confidence.get(c, 0)}" for c in (*config.CONFIDENCE_LEVELS, None)))


def main():
    """Run with the SETTINGS at the top of this file."""
    if MODE not in ("pilot", "full"):
        raise ValueError(f'MODE must be "pilot" or "full", not {MODE!r}')

    scenes = load_scenes(IMAGES_DIR)
    total_scenes = len(scenes)
    if MODE == "pilot":
        scenes = sample_pilot_scenes(scenes, n=PILOT_SCENES, seed=PILOT_SEED)
    if LIMIT:
        scenes = scenes[:LIMIT]

    backend = make_backend(
        BACKEND,
        model_name=MODEL,
        device=DEVICE,
        dtype=DTYPE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        base_url=SERVER_URL,
        api_key=SERVER_API_KEY,
        workers=SERVER_WORKERS,
        use_schema=SERVER_ENFORCE_SCHEMA,
    )
    if BACKEND == "server" and BATCH_SIZE < SERVER_WORKERS:
        print(f"Note: BATCH_SIZE ({BATCH_SIZE}) is smaller than SERVER_WORKERS ({SERVER_WORKERS}); "
              f"raise BATCH_SIZE to keep all workers busy.")

    output_name = OUTPUT_FILE or default_output_name(MODE, MODEL, BACKEND)
    output_path = os.path.join(OUTPUT_DIR or config.LABELS_DIR, output_name)

    stats = label_scenes(
        scenes, backend, output_path,
        batch_size=BATCH_SIZE,
        model_name=MODEL if BACKEND != "dry-run" else "dry-run",
        backend_name=BACKEND,
        retry_invalid=RETRY_INVALID,
    )

    if stats["processed"]:
        per_scene = stats["seconds"] / stats["processed"]
        print(f"Valid answers: {stats['valid']}/{stats['processed']}")
        print(f"Time per scene: {per_scene:.2f} s -> all {total_scenes} scenes would take "
              f"about {per_scene * total_scenes / 3600:.1f} h with these settings")
    print_distribution(load_labels(output_path))
    print(f"\nAnswers are in: {output_path}")
    print("Use this name as LABELS_FILE in compare_labels.py.")


if __name__ == "__main__":
    main()
