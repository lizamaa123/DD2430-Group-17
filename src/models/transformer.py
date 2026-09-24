import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFurniturePredictor(nn.Module):

    def __init__(self, embed_dim=384, nhead=4, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim

        # Trainable representation for the masked slot
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Projection head, outputting logits
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, mask_index):
        """Does inference for a batch of room sets with already embedded furniture/label vectors.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask_index (int): The index of the furniture item to be masked in each sample.
        Returns:
            torch.Tensor: Predicted vector of shape (embed_dim) for the masked furniture item, normalized to unit length."""

        # x shape: (batch_size, seq_len, embed_dim)
        # Replace the target index in each sample with the learnable mask token
        batch_size = x.size(0)
        x = x.clone()
        x[:, mask_index, :] = self.mask_token.expand(batch_size, -1, -1)

        # Pass continuous vectors directly through the Transformer
        out = self.transformer(x)

        # Extract the hidden vector at the masked index
        predicted_vec = self.proj_head(out[:, mask_index, :])

        # Normalize to unit sphere (essential for cosine similarity matching)
        return F.normalize(predicted_vec, p=2, dim=-1)
