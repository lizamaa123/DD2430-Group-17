# Room labelling

DeepFurniture has no room types, but Part 2 of the project compares models with and
without room context. This folder labels every DeepFurniture scene with a room type by
showing its image (one rendered view per scene) to a vision-language model (Qwen, the
same model and backends as `image_to_text/`).

## Room types

Defined in `config.py` (`ROOM_TYPES`), chosen after looking at samples of the scenes:

| Label | Notes |
|---|---|
| `bedroom`, `living_room`, `dining_room`, `office`, `kids_room` | Same names as the synthetic rooms (`room_generator.ANCHOR_RULES`) |
| `living_dining` | Open-plan room with a sofa and a dining table in view |
| `hallway` | Corridors and entrance areas, common in DeepFurniture |
| `other` | Empty or unidentifiable rooms |

There are no kitchens or bathrooms in the data, so they aren't labels. How
`living_dining`, `hallway` and `other` are used in the experiments is decided later,
when the labels are added to `deepfurniture_rooms.json`.

The model answers `{"reason": ..., "room_type": ..., "confidence": "high|medium|low"}`.
It writes the reason first, so it looks before it decides.

## Files

| File | What it does |
|---|---|
| `shrink_scene_images.py` | Copies the scene images at 768 px (about 1.3 GB instead of 13 GB) and zips them for Kaggle |
| `run_room_labelling.py` | Main runner. `MODE` is `pilot` (150 fixed scenes) or `full` (all 19,399). Resumable |
| `label_by_hand.py` | Shows the pilot scenes one at a time; label them with the number keys |
| `compare_labels.py` | Model labels vs hand labels: accuracy, precision/recall, confusion matrix, mistakes shown with images |
| `config.py`, `prompts.py`, `parsing.py`, `dataset.py` | Settings and room types; prompt and schema; answer checking; loaders |

Outputs go to `data/deepfurniture/room_labels/` (not pushed to GitHub):
`labels/<mode>_<model>_<version>.jsonl`, `hand_labels.csv`, `scene_images_768/`.

## Steps

1. **Shrink the images**: run `shrink_scene_images.py`. It writes `scene_images_768.zip`.
2. **Hand labels**: run `label_by_hand.py` and label the 150 pilot scenes (about 10 min).
   Do this before looking at the model's answers.
3. **Pilot on Kaggle** (see below) with `MODE = "pilot"`.
4. **Compare**: put the pilot's `.jsonl` in `data/deepfurniture/room_labels/labels/`, set
   `LABELS_FILE` in `compare_labels.py` and run it. If the labels aren't good enough,
   change `ROOM_TYPES` or the prompt, increase `LABEL_VERSION` in `config.py`, and run the
   pilot again.
5. **Full run on Kaggle** with `MODE = "full"`.

## Running on Kaggle

1. Upload `scene_images_768.zip` as a new Kaggle dataset (Kaggle unzips it).
2. In a notebook (GPU T4, internet on), add that dataset and get this repo's code
   (the same way as for the captioning run).
3. Install a recent transformers: `pip install -U transformers`.
4. Set in `run_room_labelling.py`:
   ```python
   MODE = "pilot"            # then "full"
   BACKEND = "transformers"
   DTYPE = "float16"         # about 10x faster on a T4
   BATCH_SIZE = 8
   IMAGES_DIR = "/kaggle/input/<your-dataset>/scene_images_768"  # the folder with the .jpg files
   OUTPUT_DIR = "/kaggle/working"
   ```
5. Run it, then download the `.jsonl` from the notebook's output.

**Continuing a stopped run on Kaggle:** `/kaggle/working` is empty in a new session.
Copy the partial `.jsonl` there first (e.g. from a dataset made of the previous output);
the scenes already in it are skipped.

The pilot prints the time per scene and an estimate for all scenes, so you can check that
the full run fits in a Kaggle session (12 h) before starting it.
