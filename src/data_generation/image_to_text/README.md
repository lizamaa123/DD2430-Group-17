# Image to text (DeepFurniture)

## What this folder is for

Our synthetic rooms are built from IKEA items, and each IKEA item is described by **text**
(name, category, short description). We turn that text into an embedding with a sentence
encoder, and the model learns from those embeddings.

DeepFurniture items have **no text**, only a picture of each item. To compare the two
datasets, both need to be described in the same way. So this code shows each DeepFurniture
picture to a vision-language model (Qwen), which looks at the image and writes a short
description of it. That description is then turned into a text that looks like an IKEA
item's text, and embedded with the same encoder.

## How it works, step by step

1. **Pick the items.** Every DeepFurniture item with a picture, or a small fixed sample of
   them for testing (the "pilot").
2. **Ask the model.** For each item, the model gets the picture and a list of instructions:
   which category it is, what kind of item it is, its colour, material, style, and so on.
   It answers in a fixed format (JSON).
3. **Check the answer.** Answers that are broken (e.g. not valid JSON, or a category that
   isn't allowed) are fixed if the problem is small, or asked again if not.
4. **Save everything.** Each answer is saved to a file as soon as it's done, so a run can be
   stopped and continued later.
5. **Turn answers into IKEA-style texts.** Each answer becomes a text with the same layout as
   the IKEA texts, in a few versions (shorter and longer).
6. **Embed the texts** with the same sentence encoder as the IKEA items.
7. **Check the quality.** Mostly for the pilot: are the answers sensible, does the model's
   style agree with DeepFurniture's labels, how different do the two datasets look?

## The files

**The files you run**, in this order:

| File | What it's for |
|---|---|
| `run_captioning.py` | **The main script.** Goes through the items, asks the model, checks the answers and saves them. |
| `build_texts.py` | Turns the saved answers into IKEA-style texts and embeds them. |
| `quality_checks.py` | Prints a report on how good the answers are, and can write a spreadsheet for checking answers by hand. |
| `show_captions.py` | Shows the item pictures with the model's answer underneath, to check by eye that the descriptions are right. |

**The helper files** (used by the ones above, not run on their own):

| File | What it's for |
|---|---|
| `config.py` | Settings shared by all files: file paths, the default model, which IKEA categories each DeepFurniture category can become, the list of styles, and example phrases shown to the model. Change these when you want to change what the model is asked. |
| `dataset.py` | Reads the data: the DeepFurniture items and their labels, the pilot sample, the IKEA items, and the answer files this code writes. |
| `prompts.py` | Writes the instructions the model gets for each item, and the exact format its answer must have. |
| `backends.py` | Actually runs the model. There are three ways: load the model directly (`transformers`), send requests to a model server (`server`), or a fake model for testing (`dry-run`). |
| `parsing.py` | Reads the model's answer, cleans it up and checks that it's usable. |

## What the model answers

For each picture the model returns something like this:

```json
{"ikea_category": "Tables & desks",
 "type": "Bedside table with 3 drawers",
 "colour": "white body, brown top",
 "material": "painted wood, metal handles",
 "details": "panelled drawer fronts, plinth base",
 "style": ["European/American"],
 "caption": "A white three-drawer bedside table with a brown wooden top.",
 "uncertain": false}
```

- **ikea_category**: which IKEA category the item belongs to. The model can only choose from
  a short list that fits the item (e.g. a DeepFurniture "table" can become "Tables & desks",
  "Bar furniture", ...). Lamps, doors, curtains, plants, decorations and appliances don't
  exist in our IKEA data, so they get a fixed name instead (e.g. "Lighting").
- **type**: what the item is, written in the short style of IKEA's descriptions. No colour or
  size here, so it can be compared with IKEA's text.
- **colour, material, details**: extra description that the IKEA texts don't have.
- **style**: the model's guess, using DeepFurniture's own style names, so it can be compared
  with the existing labels (and fill in the ~9% of items without one).
- **caption**: one free sentence describing the item.
- **uncertain**: the model says it isn't sure what the item is (e.g. several objects in one
  picture). Treat this as a hint; the model isn't very good at judging itself.

## The text versions

Each item is written in the same layout as an IKEA item. Series name and designer don't
exist for DeepFurniture, so they are left empty:

```
Product series: . Category: Tables & desks. Description: Bedside table with 3 drawers. Designer: .
```

Only the description part changes between the versions:

| Version | Description | When to use it |
|---|---|---|
| `short` | type | Comparing or mixing with IKEA (e.g. train on one dataset, test on the other). It holds the same kind of information as the IKEA texts. |
| `medium` | type + colour | Experiments within DeepFurniture only. |
| `rich` | type + colour + material + details | Experiments within DeepFurniture only. |
| `caption` | the free sentence | Experiments within DeepFurniture only. |

The longer versions contain information the IKEA texts don't have (like colour), so they
would make the two datasets look more different than they are.

## How to run it

There are four files you run: `run_captioning.py`, `build_texts.py`, `quality_checks.py` and
`show_captions.py`. Each has a **SETTINGS** block at the top. Change the settings, then run
the file (e.g. with the Run button in VS Code). The other files are only used by these four.

Extra packages needed: `transformers` and `accelerate` (to run the model directly), `openai`
(to talk to a model server), `sentence-transformers` (for the embeddings).

### 1. Check that everything works (takes seconds)

Run `run_captioning.py` as it is. The default settings use the fake model (`BACKEND = "dry-run"`),
so this only tests the code, not the model.

### 2. Pilot: describe 154 items with the real model

In `run_captioning.py`, set `BACKEND = "transformers"` (or `"server"`, see
[Where to run it](#where-to-run-it)) and run it. Keep `MODE = "pilot"`: this describes 14 items
from each of the 11 categories, always the same ones.

Then set `PROMPT_VARIANT = "short"` and run it once more. This asks only for category and
type, on the same items, to check that asking for many fields at once doesn't change the
short description.

Tip: set `LIMIT = 5` the first time, to see that the model works before running all 154.

At the end, the script prints the name of the file the answers were saved to, e.g.
`pilot_Qwen3.5-4B_full_v2.jsonl`. The other two scripts need this name.

### 3. Look at the pilot results

In `quality_checks.py`, set `CAPTIONS_FILE` to the full-prompt file and `COMPARE_FILE` to the
short-prompt file, then run it. It prints a report and writes a spreadsheet
(`..._grading.csv`) with every answer, so you can mark right and wrong while looking at the
images.

To look at the answers next to the pictures, set `CAPTIONS_FILE` in `show_captions.py` and
run it. It shows 6 items per window; close the window to see the next ones. Answers in red
need a closer look (unusable, uncertain or flagged by the checks). With `ONLY = "problems"`
or `CATEGORY = "lamp"` you can show just those items.

### 4. Make the texts and embeddings

In `build_texts.py`, set `CAPTIONS_FILE` and run it. It writes a CSV with all the texts (good
for reading through) and one embeddings file per text version.

To see how close the new embeddings are to the IKEA ones, set `PROBE_VARIANT = "short"` in
`quality_checks.py` and run it again.

### 5. Full run on all items

In `run_captioning.py`, set `MODE = "full"` and run it. This is the long one, so it's best done
on a GPU (see below). If it stops, just run it again: it continues where it stopped.

## Where to run it

The default model (Qwen3.5-4B) needs about 9 GB of memory, more than an 8 GB Mac has.

- **On a GPU** (Colab, Kaggle, KTH): use `BACKEND = "transformers"`. For the full run it's
  faster to start a vLLM server there (`vllm serve Qwen/Qwen3.5-4B`) and use
  `BACKEND = "server"` with e.g. `BATCH_SIZE = 32` and `SERVER_WORKERS = 16`.
- **On a Mac**, either:
  - test with a smaller model: `BACKEND = "transformers"` and `MODEL = "Qwen/Qwen3.5-0.8B"`, or
  - load a compressed (4-bit) Qwen3.5 model in LM Studio and start its server, then use
    `BACKEND = "server"`, `SERVER_URL = "http://localhost:1234/v1"` and set `MODEL` to the
    name LM Studio shows.

## Good to know

- **Stopping and continuing**: if a run stops, run the file again with the same settings.
  Items that are already done are skipped.
- **Where the results go**: `data/deepfurniture/image_to_text/` (not pushed to GitHub).
- **Changing the instructions**: increase `PROMPT_VERSION` in `config.py`. The version is
  part of the file name, so answers from old and new instructions never get mixed up.
- **The time estimate**: at the end of each run, the script prints how long the full dataset
  would take with the same settings. Use the pilot's estimate to decide where to run the
  full run.
