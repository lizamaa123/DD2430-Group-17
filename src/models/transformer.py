import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

class FurnitureEmbeddingPredictor(nn.Module):

    def __init__(self, embed_dim=384, hid_dim = 256, nhead=4, num_layers=3, device = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Loaded instance on device: {self.device}")
        self.to(self.device)

        # Trainable representation for the masked slot
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim, device=self.device))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hid_dim,
            batch_first=True,
            device=self.device
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Projection head, outputting logits
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, device=self.device),
        )

    def forward(self, x, valid_mask, mask_index):
        """Does inference for a batch of room sets with already embedded furniture/label vectors.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask_index (int): The index of the furniture item to be masked in each sample.
        Returns:
            torch.Tensor: Predicted vector of shape (embed_dim) for the masked furniture item, normalized to unit length."""

        # x shape: (batch_size, seq_len, embed_dim)
        # Replace the target index in each sample with the learnable mask token
        batch_size = x.size(0)

        batch_indices = torch.arange(batch_size, device=x.device)
        x = x.clone()

        # Replace one item in each room with MASK
        x[batch_indices, mask_index, :] = self.mask_token[0, 0, :]

        # True means "ignore this position" for PyTorch Transformer
        padding_mask = ~valid_mask

        # Transformer
        out = self.transformer(x, src_key_padding_mask=padding_mask)

        # Extract the hidden vector at the masked index
        predicted_vec = self.proj_head(out[batch_indices, mask_index, :])

        # Normalize to unit sphere (essential for cosine similarity matching)
        return F.normalize(predicted_vec, p=2, dim=-1)
    
    def fit(self, train_loader, val_loader=None, epochs=20, debug_prints=False):
        """Trains the transformer on positive pairs ONLY (currently) with cosine similarity and AdamW optimiser."""

        print(f"Training with device: {self.device}")
        self.to(self.device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        cos_loss = torch.nn.CosineEmbeddingLoss()

        # logging
        train_batch_losses = []

        pbar = tqdm.tqdm(range(epochs))

        for epoch in pbar:

            # Put PyTorch model into training mode
            self.train()

            epoch_loss = 0

            for batch in train_loader:

                # x shape: [batch_size, room_length, 384]
                x = batch["x"].to(self.device)

                # True = real furniture, False = PAD
                valid_mask = batch["valid_mask"].to(self.device)

                batch_size = x.size(0)

                batch_indices = torch.arange(
                    batch_size,
                    device=self.device
                )

                # Pick one REAL furniture item per room to hide.
                # PAD positions have prob value 0, so they cannot be selected.
                mask_idx = torch.multinomial(valid_mask.float(),1).squeeze(1)

                # Save the true furniture embedding before masking.
                true = x[batch_indices, mask_idx,:]

                # 1 = positive pair:
                # prediction should be similar to the true embedding.
                pair_flags = torch.ones(batch_size, device=self.device)

                optimizer.zero_grad()

                # Transformer replaces the chosen real item with MASK.
                # valid_mask tells the Transformer which positions are PAD.
                pred = self.forward(x,valid_mask,mask_idx)

                loss = cos_loss(
                    pred,
                    true,
                    pair_flags
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                train_batch_losses.append(
                    loss.item()
                )

                if debug_prints:
                    print(f"batch loss: {loss.item()}")

            epoch_loss /= len(train_loader)

            print(
                f"Average epoch training loss: "
                f"{epoch_loss:.4f}"
            )

        # Plot training loss for every batch
        fig = plt.figure()

        plt.plot(
            np.arange(len(train_batch_losses)),
            train_batch_losses
        )

        plt.suptitle("Batch training loss")
        plt.xlabel("Batch")
        plt.ylabel("Cosine loss")
        plt.show()

"""
x
shape = [B,L,384]

valid_mask
shape = [B,L]

mask_idx
shape = [B]

"""
