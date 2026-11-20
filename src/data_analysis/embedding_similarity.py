"""
Study how similar items are in the SBERT embedding space.

Shows what a given cosine similarity "means": for each similarity level
(e.g. 0.9-0.95, 0.5-0.6, ...) it shows example item pairs, with their
pictures (DeepFurniture) and texts. It also prints and plots how the
similarities are distributed, for random pairs and for each item's
nearest neighbour.

How to use: set the SETTINGS below and run this file.
- DATASET picks the items. Set DATASET_B to compare two datasets with
  each other (e.g. IKEA vs DeepFurniture "short", which share the
  embedding space); pairs are then always one item from each.
- One window opens with the distributions, then one window per similarity
  level with example pairs. Close a window to see the next one. Set
  SAVE_FIGURES = True to save them as .png files instead.

Datasets:
- "df_short", "df_caption": DeepFurniture items, embeddings from
  image_to_text/build_texts.py.
- "ikea": IKEA catalogue, embeddings from data_generation/item_encoder.py
  (data/item_embeddings.pt). IKEA items have no pictures, only text.
"""

import os
import random
import sys
import textwrap

# Make the repo root importable, so this file can be run directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


# ===========================================================================
# SETTINGS - change these before running
# ===========================================================================

DATASET = "df_short"  # "df_short", "df_caption" or "ikea"
DATASET_B = None  # None: pairs within DATASET. Otherwise another dataset name.

# Only use items from these categories (None = all). DeepFurniture uses its
# own categories ("chair#stool", "lamp", ...), IKEA its catalogue categories.
CATEGORIES = None  # e.g. ["chair#stool", "sofa"]
# "all": any pair. "same": both items in the same category. "different":
# items in different categories. Only used when DATASET_B is None.
PAIR_CATEGORY = "all"

# Similarity levels to show examples for: each pair of neighbouring edges
# is one level. The top level (>= 0.999) is "identical text" in practice.
BIN_EDGES = [-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999, 1.01]
EXAMPLES_PER_BIN = 6  # pairs shown per level
SHOW_LEVELS = None  # None = every level; or a list of lower edges, e.g. [0.5, 0.9]

# Pairs are sampled by taking N_ANCHORS random items and comparing each with
# every other item. More anchors = better estimates, more memory
# (N_ANCHORS x number of items floats).
N_ANCHORS = 4000

SHOW_IMAGES = True  # False: print the pairs only, no example windows
SAVE_FIGURES = False  # True: save .png files to data/analysis/embedding_similarity/
SEED = 42

# ===========================================================================

DF_RUN = "full_Qwen3.5-2B_full_v2"
DF_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "image_to_text")
DF_IMAGES_DIR = os.path.join(REPO_ROOT, "data", "deepfurniture", "data", "furnitures")
IKEA_CSV_PATH = os.path.join(REPO_ROOT, "data", "processed_ikea_data.csv")
IKEA_EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "data", "item_embeddings.pt")
FIGURES_DIR = os.path.join(REPO_ROOT, "data", "analysis", "embedding_similarity")

TEXT_WIDTH = 45  # characters per line under each picture


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def ikea_item_text(row):
    """
    The IKEA item text, built exactly like item_encoder.build_item_text
    (copied so this file does not need sentence_transformers).

    Args:
        row: One row of processed_ikea_data.csv.

    Returns:
        The text that was embedded.
    """
    def field(name):
        return str(row[name]) if pd.notna(row[name]) else ""

    return (
        f"Product series: {field('name')}. "
        f"Category: {field('category')}. "
        f"Description: {field('short_description')}. "
        f"Designer: {field('designer')}."
    )


def short_text(text):
    """Drop the empty "Product series: ." / "Designer: ." parts of DF texts."""
    return text.replace("Product series: . ", "").replace(" Designer: .", "")


def load_dataset(name):
    """
    Load the items, their embeddings, texts, categories and pictures.

    Args:
        name: "df_short", "df_caption" or "ikea".

    Returns:
        Dict with "name", "ids" (list), "emb" (float32 tensor, one
        normalized row per item), "texts", "categories" and "images"
        (picture paths, or None for items without one), all in the same
        order.
    """
    if name in ("df_short", "df_caption"):
        variant = name.split("_", 1)[1]
        emb_path = os.path.join(DF_DIR, f"item_embeddings_{variant}_{DF_RUN}.pt")
        table = pd.read_csv(os.path.join(DF_DIR, f"item_texts_{DF_RUN}.csv"))
        info = {
            int(r.furniture_id): (short_text(getattr(r, f"text_{variant}")), r.df_category)
            for r in table.itertuples()
        }
        image_of = lambda item_id: os.path.join(DF_IMAGES_DIR, f"{item_id}.jpg")  # noqa: E731
    elif name == "ikea":
        emb_path = IKEA_EMBEDDINGS_PATH
        table = pd.read_csv(IKEA_CSV_PATH)
        info = {int(row["item_id"]): (ikea_item_text(row), row["category"]) for _, row in table.iterrows()}
        image_of = lambda item_id: None  # noqa: E731
    else:
        raise ValueError(f'Unknown dataset {name!r}; use "df_short", "df_caption" or "ikea"')

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"No embeddings at {emb_path}. Run "
            + ("src/data_generation/item_encoder.py" if name == "ikea" else "image_to_text/build_texts.py")
            + " first."
        )
    embeddings = torch.load(emb_path)

    ids = [i for i in embeddings if i in info]
    if CATEGORIES is not None:
        ids = [i for i in ids if info[i][1] in CATEGORIES]
    if not ids:
        raise ValueError(f"No items left in {name} after the CATEGORIES filter")

    emb = torch.stack([embeddings[i].float() for i in ids])
    emb = torch.nn.functional.normalize(emb, dim=1)  # should already be normalized
    return {
        "name": name,
        "ids": ids,
        "emb": emb,
        "texts": [info[i][0] for i in ids],
        "categories": np.array([info[i][1] for i in ids]),
        "images": [image_of(i) for i in ids],
    }


# ---------------------------------------------------------------------------
# Similarities
# ---------------------------------------------------------------------------

def anchor_similarities(a, b, same_dataset, rng):
    """
    Cosine similarity of N_ANCHORS random items of `a` to every item of `b`.

    Args:
        a, b: Datasets from load_dataset.
        same_dataset: True when a and b are the same, so an item is never
            paired with itself.
        rng: numpy random generator.

    Returns:
        (anchors, sims, valid): anchor indices into a, the
        (n_anchors x len(b)) similarity matrix, and a boolean matrix of
        the same shape marking the pairs that count (not the item itself,
        PAIR_CATEGORY respected).
    """
    n_anchors = min(N_ANCHORS, len(a["ids"]))
    anchors = rng.choice(len(a["ids"]), size=n_anchors, replace=False)
    sims = (a["emb"][anchors] @ b["emb"].T).clamp(-1, 1).numpy()

    valid = np.ones_like(sims, dtype=bool)
    if same_dataset:
        valid[np.arange(n_anchors), anchors] = False
        same_cat = a["categories"][anchors][:, None] == b["categories"][None, :]
        if PAIR_CATEGORY == "same":
            valid &= same_cat
        elif PAIR_CATEGORY == "different":
            valid &= ~same_cat
        elif PAIR_CATEGORY != "all":
            raise ValueError(f'PAIR_CATEGORY must be "all", "same" or "different", not {PAIR_CATEGORY!r}')
    return anchors, sims, valid


def nearest_neighbour_similarities(a, b, same_dataset, chunk=1024):
    """
    For every item of `a`, the similarity to its most similar item in `b`.

    Args:
        a, b: Datasets from load_dataset.
        same_dataset: True when a and b are the same (the item itself is
            skipped).
        chunk: Rows compared at once, to keep memory low.

    Returns:
        (best_sims, best_index): numpy arrays of length len(a).
    """
    best_sims, best_index = [], []
    for start in range(0, len(a["ids"]), chunk):
        sims = a["emb"][start:start + chunk] @ b["emb"].T
        if same_dataset:
            rows = torch.arange(sims.shape[0])
            sims[rows, rows + start] = -2.0
        values, index = sims.max(dim=1)
        best_sims.append(values.clamp(-1, 1))
        best_index.append(index)
    return torch.cat(best_sims).numpy(), torch.cat(best_index).numpy()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def level_label(low, high):
    """Readable name of one similarity level."""
    if high > 1.0:
        return f">= {low:g}"
    if low <= -1.0:
        return f"< {high:g}"
    return f"{low:g} - {high:g}"


def print_summary(a, b, pair_sims, nn_sims, same_dataset):
    """Print sizes, duplicate texts and similarity percentiles."""
    title = a["name"] if same_dataset else f"{a['name']} vs {b['name']}"
    print(f"\n=== Cosine similarity: {title} ===")
    for d in ([a] if same_dataset else [a, b]):
        unique = len(set(d["texts"]))
        print(f"{d['name']}: {len(d['ids'])} items, {unique} unique texts "
              f"({len(d['ids']) - unique} items share their text with another item)")

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nRandom pairs ({len(pair_sims):,} sampled, PAIR_CATEGORY={PAIR_CATEGORY!r}):")
    print("  mean %.3f | " % pair_sims.mean()
          + " | ".join(f"p{p} {v:.3f}" for p, v in zip(percentiles, np.percentile(pair_sims, percentiles))))
    print("Nearest neighbour of each item" + ("" if same_dataset else f" of {a['name']} in {b['name']}") + ":")
    print("  mean %.3f | " % nn_sims.mean()
          + " | ".join(f"p{p} {v:.3f}" for p, v in zip(percentiles, np.percentile(nn_sims, percentiles))))
    print(f"  {np.mean(nn_sims >= 0.999):.1%} of items have an identical-text neighbour (sim >= 0.999)")

    print("\nShare of pairs per similarity level:")
    print(f"  {'level':>14} {'random pairs':>13} {'nearest nb.':>12}")
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        in_pairs = np.mean((pair_sims >= low) & (pair_sims < high))
        in_nn = np.mean((nn_sims >= low) & (nn_sims < high))
        print(f"  {level_label(low, high):>14} {in_pairs:>12.2%} {in_nn:>11.2%}")


def plot_distributions(a, b, pair_sims, nn_sims, same_cat_sims, same_dataset):
    """Histograms of random-pair and nearest-neighbour similarities."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-0.2, 1.0, 121)

    ax = axes[0]
    if same_cat_sims is not None:
        ax.hist(same_cat_sims[0], bins=bins, density=True, alpha=0.6, label="same category")
        ax.hist(same_cat_sims[1], bins=bins, density=True, alpha=0.6, label="different category")
        ax.legend()
    else:
        ax.hist(pair_sims, bins=bins, density=True, alpha=0.8)
    ax.set_title("Random pairs")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")

    ax = axes[1]
    ax.hist(nn_sims, bins=bins, alpha=0.8)
    ax.set_yscale("log")  # most items often have an identical-text neighbour
    ax.set_title("Nearest neighbour of each item" + ("" if same_dataset else f" ({b['name']})"))
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("items (log scale)")

    title = a["name"] if same_dataset else f"{a['name']} vs {b['name']}"
    fig.suptitle(f"Cosine similarity in the embedding space: {title}")
    fig.tight_layout()
    finish_figure(fig, f"distribution_{title.replace(' ', '_')}")


def sample_pairs(sims, valid, low, high, rng):
    """Up to EXAMPLES_PER_BIN random (anchor row, b index) pairs in one level."""
    rows, cols = np.nonzero(valid & (sims >= low) & (sims < high))
    if len(rows) == 0:
        return []
    pick = rng.choice(len(rows), size=min(EXAMPLES_PER_BIN, len(rows)), replace=False)
    return [(rows[k], cols[k]) for k in pick]


def show_item(ax, dataset, index):
    """Draw one item's picture (or a blank box) with its id and text below."""
    path = dataset["images"][index]
    if path and os.path.exists(path):
        ax.imshow(Image.open(path).convert("RGB"))
    else:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.5, "no picture", ha="center", va="center", color="grey", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    text = f"{dataset['name']} {dataset['ids'][index]}\n" + textwrap.fill(dataset["texts"][index], TEXT_WIDTH)
    ax.set_xlabel(text, fontsize=7.5, loc="left")


def show_level(a, b, anchors, sims, pairs, label):
    """Print the example pairs of one level and show them as a figure."""
    print(f"\n--- Similarity {label}: {len(pairs)} example pairs ---")
    for row, col in pairs:
        i = anchors[row]
        print(f"  {sims[row, col]:.3f}  [{a['ids'][i]}] {a['texts'][i]}")
        print(f"         [{b['ids'][col]}] {b['texts'][col]}")
    if not SHOW_IMAGES or not pairs:
        return

    # Two pairs per figure row: picture A, picture B, gap, picture A, picture B.
    n_rows = (len(pairs) + 1) // 2
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 4.8 * n_rows), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.15, 1, 1]}, squeeze=False)
    for ax in axes[:, 2]:
        ax.axis("off")
    for k, (row, col) in enumerate(pairs):
        left = axes[k // 2, 0 if k % 2 == 0 else 3]
        right = axes[k // 2, 1 if k % 2 == 0 else 4]
        show_item(left, a, anchors[row])
        show_item(right, b, col)
        left.set_title(f"cos = {sims[row, col]:.3f}", fontsize=10, loc="left", fontweight="bold")
    if len(pairs) % 2 == 1:
        axes[-1, 3].axis("off")
        axes[-1, 4].axis("off")

    fig.suptitle(f"Pairs with cosine similarity {label}", fontsize=13)
    finish_figure(fig, f"pairs_{label.replace(' ', '').replace('>=', 'ge').replace('<', 'lt')}")


def finish_figure(fig, name):
    """Save the figure (SAVE_FIGURES) or show it."""
    if SAVE_FIGURES:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
    else:
        plt.show()


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    a = load_dataset(DATASET)
    same_dataset = DATASET_B is None or DATASET_B == DATASET
    b = a if same_dataset else load_dataset(DATASET_B)

    anchors, sims, valid = anchor_similarities(a, b, same_dataset, rng)
    pair_sims = sims[valid]
    nn_sims, _ = nearest_neighbour_similarities(a, b, same_dataset)

    # Within- vs across-category split for the histogram (same dataset, all pairs).
    same_cat_sims = None
    if same_dataset and PAIR_CATEGORY == "all":
        same_cat = a["categories"][anchors][:, None] == b["categories"][None, :]
        same_cat_sims = (sims[valid & same_cat], sims[valid & ~same_cat])

    print_summary(a, b, pair_sims, nn_sims, same_dataset)
    plot_distributions(a, b, pair_sims, nn_sims, same_cat_sims, same_dataset)

    # Highest similarity first.
    levels = list(zip(BIN_EDGES[:-1], BIN_EDGES[1:]))[::-1]
    for low, high in levels:
        if SHOW_LEVELS is not None and low not in SHOW_LEVELS:
            continue
        pairs = sample_pairs(sims, valid, low, high, rng)
        show_level(a, b, anchors, sims, pairs, level_label(low, high))


if __name__ == "__main__":
    main()
