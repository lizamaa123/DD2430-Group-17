import json
import torch
from torch.utils.data import Dataset, DataLoader


ROOMS_PATH = "data/synthetic_rooms.json"
EMBEDDINGS_PATH = "data/item_embeddings.pt"

EMBED_DIM = 384


class RoomDataset(Dataset):

    def __init__(self, split="train"):

        # Load synthetic rooms
        with open(ROOMS_PATH, "r") as f:
            rooms = json.load(f)

        # Keep only train / val / test rooms
        self.rooms = [room for room in rooms if room["split"] == split]

        # Load Sentence-BERT embeddings
        self.item_embeddings = torch.load(EMBEDDINGS_PATH, weights_only=False)

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx):

        room = self.rooms[idx]

        item_ids = room["item_ids"]

        # Convert each furniture ID into its 384-D embedding
        embeddings = torch.stack([self.item_embeddings[item_id] for item_id in item_ids])

        return {
            "embeddings": embeddings,
            "item_ids": torch.tensor(
                item_ids,
                dtype=torch.long
            ),
            "room_type": room["room_type"],
        }

"""As all rooms do not contain the same amount of furniture, pytorch will not be able to stack diffrent enumarated rooms into one vector eg [3,384],[5,384],[4,384]
This tells us room 1 contain 3 items and we havethe size of each vector 384.
"""
def collate_rooms(batch):

    batch_size = len(batch)

    # Find the largest room in this batch
    max_room_size = max(example["embeddings"].shape[0] for example in batch)

    # Tensor for all room embeddings
    x = torch.zeros(batch_size, max_room_size, EMBED_DIM)

    # False = padding
    # True = actual furniture
    valid_mask = torch.zeros(batch_size, max_room_size, dtype=torch.bool)

    item_ids = torch.full((batch_size, max_room_size), fill_value=-1, dtype=torch.long)

    room_types = []

    for i, example in enumerate(batch):

        embeddings = example["embeddings"]
        ids = example["item_ids"]

        room_len = embeddings.shape[0]

        # Add actual furniture
        x[i, :room_len] = embeddings

        # Mark real furniture positions
        valid_mask[i, :room_len] = True

        # Save item IDs
        item_ids[i, :room_len] = ids

        room_types.append(example["room_type"])

    return {
        "x": x,
        "valid_mask": valid_mask,
        "item_ids": item_ids,
        "room_types": room_types,
    }

if __name__ == "__main__":

    dataset = RoomDataset(split="train")

    print("Number of rooms:", len(dataset))

    room = dataset[0]

    print("\nSingle room:")
    print("Embeddings shape:", room["embeddings"].shape)
    print("Item IDs:", room["item_ids"])
    print("Room type:", room["room_type"])

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_rooms,
    )

    batch = next(iter(loader))

    print("\nBatch:")
    print("X shape:", batch["x"].shape)
    print("Valid mask shape:", batch["valid_mask"].shape)
    print("Item IDs shape:", batch["item_ids"].shape)

    print("\nFirst room valid mask:")
    print(batch["valid_mask"][0])

    print("\nFirst room item IDs:")
    print(batch["item_ids"][0])