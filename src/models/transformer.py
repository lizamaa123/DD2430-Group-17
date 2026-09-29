import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

class SemanticFurniturePredictor(nn.Module):

    def __init__(self, embed_dim=128, hid_dim = 256, nhead=4, num_layers=3, device = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.to(self.device)

        # Trainable representation for the masked slot
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hid_dim,
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


    def train(self, X_train: torch.Tensor, X_val: torch.Tensor, batch_size = 100, epochs = 20, debug_prints = False):
        """Trains the transformer on positive pairs ONLY (currently) with cosine similarity and AdamW optimiser.
        Assuming X_train and X_val to have shape [nbr_datapoints, sequence_length, embedding_dim]"""
        
        
        print(f"Training with device: {self.device}")
        self.to(self.device)
        X_train = X_train.to(self.device)
        X_val = X_val.to(self.device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        cos_loss = torch.nn.CosineEmbeddingLoss()

    
        num_minibatches_train = math.ceil(X_train.shape[0]/batch_size)
        num_minibatches_val = math.ceil(X_val.shape[0]/batch_size)

        # logging
        train_batch_losses = []

        pbar = tqdm.tqdm(range(epochs))
        for i in pbar:
            print(f"\nEpoch {i} ============")
            epoch_loss = 0
            
            perm = torch.randperm(X_train.shape[0], device=self.device)
            X_train = X_train[perm]
            
            for j in range(num_minibatches_train):
                X_batch = X_train[j*batch_size:min((j+1)*batch_size, X_train.shape[0])]
                cur_batch_size = X_batch.shape[0]

                optimizer.zero_grad()

                # Create one mask for each room, room label at index 1
                mask_idx = torch.randint(low = 1, high = X_train.shape[1], size=(cur_batch_size,), device=self.device) 
                # 1 for positive pair, -1 for negative pair
                # TODO: Add negative pairs
                pair_flags = torch.ones_like(mask_idx, device=self.device, dtype=torch.float32)

                pred = self.forward(X_batch, mask_idx)
                true = X_batch[torch.arange(cur_batch_size), mask_idx, :]
                loss: torch.Tensor = cos_loss(true, pred, pair_flags) # (batch_size, embedding_dim)
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if debug_prints: print(f"batch loss: {loss.item()}")
                train_batch_losses.append(loss.item())
            epoch_loss /= num_minibatches_train
            print(f"Average epoch training loss: {epoch_loss}")

        fig = plt.figure()
        plt.plot(np.arange(len(train_batch_losses)), train_batch_losses)
        plt.suptitle("Batch training loss")
        plt.show()




                






        
