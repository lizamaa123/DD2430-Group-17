import sys
from pathlib import Path

from torch.utils.data import DataLoader

# Find the project root
ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "src" / "data_generation"))

from room_dataset import RoomDataset, collate_rooms
from transformer import FurnitureEmbeddingPredictor


BATCH_SIZE = 128
EPOCHS = 10


def main():

    # Load the rooms and their already-created Sentence-BERT embeddings
    train_dataset = RoomDataset(split="train")

    # Group rooms into batches and pad shorter rooms
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_rooms
    )

    # Create the Transformer
    model = FurnitureEmbeddingPredictor()

    # Send the encoded room batches to the Transformer for training
    model.fit(
        train_loader,
        epochs=EPOCHS
    )


if __name__ == "__main__":
    main()