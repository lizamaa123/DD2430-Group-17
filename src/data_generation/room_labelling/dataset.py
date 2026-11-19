"""
Loading data for room labelling: the scenes to label (one image each), the
pilot sample, the label files run_room_labelling.py writes and the hand
labels from label_by_hand.py.
"""

import csv
import json
import os
import random

from src.data_generation.room_labelling import config


def load_excluded_scenes(path=config.EXCLUDED_SCENES_PATH):
    """
    Scene ids to leave out (duplicates), one per line, "#" starts a comment.

    Args:
        path: Path to deepfurniture_excluded_scenes.txt.

    Returns:
        Set of scene ids. Empty if the file doesn't exist.
    """
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}


def default_images_dir():
    """The small images if shrink_scene_images.py has been run, else the originals."""
    return config.SMALL_IMAGES_DIR if os.path.isdir(config.SMALL_IMAGES_DIR) else config.SCENES_DIR


def load_scenes(images_dir=None, excluded_path=config.EXCLUDED_SCENES_PATH):
    """
    Every scene that has an image and isn't excluded.

    Two folder layouts are understood, so the same code works locally and
    on Kaggle:
    - flat: <images_dir>/<scene_id>.jpg (from shrink_scene_images.py)
    - original: <images_dir>/<scene_id>/image.jpg

    Args:
        images_dir: Folder with the images. None picks default_images_dir().
        excluded_path: Excluded-scenes file. Missing file = nothing excluded
            (on Kaggle the small images are already filtered).

    Returns:
        List of dicts sorted by scene_id, each with "scene_id" and
        "image_path".
    """
    images_dir = images_dir or default_images_dir()
    excluded = load_excluded_scenes(excluded_path)

    scenes = []
    for name in os.listdir(images_dir):
        if name.endswith(".jpg"):
            scene_id, image_path = name[:-4], os.path.join(images_dir, name)
        else:
            scene_id, image_path = name, os.path.join(images_dir, name, "image.jpg")
            if not os.path.exists(image_path):
                continue
        if scene_id not in excluded:
            scenes.append({"scene_id": scene_id, "image_path": image_path})

    if not scenes:
        raise FileNotFoundError(f"No scene images found in {images_dir}")
    return sorted(scenes, key=lambda scene: scene["scene_id"])


def sample_pilot_scenes(scenes, n=150, seed=42):
    """
    A fixed random sample for the pilot. The same seed always gives the same
    scenes (also on Kaggle, since load_scenes sorts them), so models,
    prompts and hand labels are compared on exactly the same images.

    Args:
        scenes: Output of load_scenes.
        n: Number of scenes.
        seed: Random seed.

    Returns:
        List of scene dicts.
    """
    return random.Random(seed).sample(scenes, min(n, len(scenes)))


def load_labels(path, valid_only=False):
    """
    Read a label file written by run_room_labelling.py (one JSON record per
    line). If a scene was labelled more than once (e.g. retried), the last
    record wins.

    Args:
        path: Path to the .jsonl file.
        valid_only: Drop records whose answer couldn't be used.

    Returns:
        Dict scene_id -> record dict. Empty if the file doesn't exist yet.
    """
    records = {}
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["scene_id"]] = record
    if valid_only:
        records = {k: r for k, r in records.items() if r["valid"]}
    return records


def find_labels_file(name):
    """
    Full path of a label file given just its name (as printed at the end of
    run_room_labelling.py). A full path is also accepted.

    Args:
        name: File name in config.LABELS_DIR, or a full path.

    Returns:
        The full path.

    Raises:
        FileNotFoundError: listing the label files that do exist.
    """
    path = name if os.path.isabs(name) else os.path.join(config.LABELS_DIR, name)
    if os.path.exists(path):
        return path
    available = sorted(f for f in os.listdir(config.LABELS_DIR) if f.endswith(".jsonl")) \
        if os.path.isdir(config.LABELS_DIR) else []
    raise FileNotFoundError(
        f"No label file {name!r} in {config.LABELS_DIR}. "
        f"Available: {', '.join(available) if available else 'none yet (run run_room_labelling.py first)'}"
    )


def load_hand_labels(path=config.HAND_LABELS_PATH):
    """
    Read the hand labels written by label_by_hand.py.

    Args:
        path: Path to hand_labels.csv.

    Returns:
        Dict scene_id -> room type (str). Scenes marked "skip" are left out.
        Empty if the file doesn't exist yet.
    """
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["scene_id"]] = row["room_type"]
    return {k: v for k, v in labels.items() if v != "skip"}
