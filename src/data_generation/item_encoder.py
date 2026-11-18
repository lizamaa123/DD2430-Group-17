from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed_ikea_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "item_embeddings.pt"

MODEL_NAME = "all-MiniLM-L6-v2"


def build_item_text(row):
    """
    Convert one IKEA catalogue item into one textual description. We need this for SentenceBert to work
    """

    name = str(row["name"]) if pd.notna(row["name"]) else ""
    category = str(row["category"]) if pd.notna(row["category"]) else ""
    description = (
        str(row["short_description"])
        # In case we se NaN
        if pd.notna(row["short_description"])
        else ""
    )
    designer = str(row["designer"]) if pd.notna(row["designer"]) else ""

    return (
        f"Product series: {name}. "
        f"Category: {category}. "
        f"Description: {description}. "
        f"Designer: {designer}."
    )


def load_item_catalog(data_path=DATA_PATH):
    """Return catalog IDs and descriptions in the same order."""
    df = pd.read_csv(data_path)
    df["item_id"] = df["item_id"].astype(int)
    item_ids = df["item_id"].tolist()
    descriptions = df.apply(build_item_text, axis=1).tolist()
    return item_ids, descriptions


def load_aligned_embeddings(item_ids, embeddings_path=OUTPUT_PATH):
    """Load saved embeddings as a matrix aligned with ``item_ids``."""
    saved_embeddings = torch.load(embeddings_path, weights_only=False)
    missing_ids = [item_id for item_id in item_ids if item_id not in saved_embeddings]
    if missing_ids:
        raise KeyError(
            f"Embeddings are missing for {len(missing_ids)} catalog items; "
            f"first missing ID: {missing_ids[0]}"
        )
    return torch.stack([saved_embeddings[item_id] for item_id in item_ids])


def create_embeddings():
    df = pd.read_csv(DATA_PATH)

    # Make sure item IDs are integers
    df["item_id"] = df["item_id"].astype(int)

    # Build one text description per furniture item
    texts = df.apply(build_item_text, axis=1).tolist()

    print(f"Encoding {len(texts)} IKEA items...")

    # 384-dimensional SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True, 
        show_progress_bar=True,
    )

    # item_id -> embedding
    item_embeddings = {
        int(item_id): embedding.cpu()
        for item_id, embedding in zip(df["item_id"], embeddings)
    }

    torch.save(item_embeddings, OUTPUT_PATH)

    print(f"Saved embeddings to {OUTPUT_PATH}")
    print(f"Number of items: {len(item_embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    create_embeddings()
