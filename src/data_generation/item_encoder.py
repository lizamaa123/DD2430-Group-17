import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/processed_ikea_data.csv"
OUTPUT_PATH = "data/item_embeddings.pt"

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