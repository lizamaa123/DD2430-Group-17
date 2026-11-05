import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
from sentence_transformers import SentenceTransformer


def calculate_running_average(losses, window_size=50):
    """Computes a rolling average over the loss list using up to window_size points."""
    n = len(losses)
    if n == 0:
        return np.array([])
    cumsum = np.cumsum(losses)
    w = min(window_size, n)
    smooth = np.zeros(n)
    smooth[:w] = cumsum[:w] / np.arange(1, w + 1)
    smooth[w:] = (cumsum[w:] - cumsum[:-w]) / w
    return smooth


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
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=20,
        debug_prints=False,
        lr=2e-5,
        live_plot=True,
        plot_every=1,
    ):
        """Trains the transformer on positive pairs ONLY (currently) with cosine similarity and AdamW optimiser.

        Args:
            train_loader: DataLoader yielding batches of dicts with 'x' and 'valid_mask'.
            val_loader: Optional validation DataLoader.
            epochs (int): Number of training epochs.
            debug_prints (bool): If True, prints batch-level losses.
            lr (float): Learning rate for AdamW optimizer.
            live_plot (bool): If True, displays a live updating matplotlib chart of batch loss during training.
            plot_every (int): Frequency (in batches) at which the live chart updates.
        """

        print(f"Training with device: {self.device}")
        self.to(self.device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        cos_loss = torch.nn.CosineEmbeddingLoss()

        # logging
        train_batch_losses = []

        # Setup live plotting if requested
        is_notebook = False
        display_handle = None
        fig = None
        ax = None
        loss_line = None
        smooth_line = None

        if live_plot:
            try:
                from IPython import get_ipython
                from IPython.display import display

                ip = get_ipython()
                if ip is not None and (
                    ip.__class__.__name__ == "ZMQInteractiveShell"
                    or "IPKernelApp" in getattr(ip, "config", {})
                ):
                    is_notebook = True
            except Exception:
                is_notebook = False

            plt.ion()
            fig, ax = plt.subplots(figsize=(9, 5))
            (loss_line,) = ax.plot(
                [], [], label="Batch Loss", color="royalblue", alpha=0.5, linewidth=1
            )
            (smooth_line,) = ax.plot(
                [], [], label="Running Avg (50 batches)", color="crimson", linewidth=1.8
            )
            ax.set_title("Training Loss (Batch)")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Cosine Loss")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right")
            fig.tight_layout()

            if is_notebook:
                try:
                    display_handle = display(fig, display_id=True)
                except Exception:
                    display_handle = None
            else:
                try:
                    plt.show(block=False)
                except Exception:
                    pass

        def _update_live_chart(epoch_idx: int, current_loss: float):
            if fig is None or not plt.fignum_exists(fig.number):
                return
            n_batches = len(train_batch_losses)
            if n_batches == 0:
                return
            x_data = np.arange(1, n_batches + 1)
            loss_line.set_data(x_data, train_batch_losses)

            smooth_vals = calculate_running_average(train_batch_losses, window_size=50)
            smooth_line.set_data(x_data, smooth_vals)

            ax.relim()
            ax.autoscale_view()
            ax.set_title(
                f"Batch Training Loss - Epoch {epoch_idx + 1}/{epochs} | "
                f"Batch {n_batches} | Loss: {current_loss:.4f}"
            )

            if is_notebook and display_handle is not None:
                display_handle.update(fig)
            else:
                try:
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                except Exception:
                    pass

        pbar = tqdm.tqdm(range(epochs))

        try:
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
                    batch_loss_val = loss.item()
                    epoch_loss += batch_loss_val
                    train_batch_losses.append(batch_loss_val)

                    if live_plot and (len(train_batch_losses) % plot_every == 0):
                        _update_live_chart(epoch, batch_loss_val)

                    if debug_prints:
                        print(f"batch loss: {batch_loss_val}")

                epoch_loss /= len(train_loader)

                print(
                    f"Average epoch training loss: "
                    f"{epoch_loss:.4f}"
                )

        finally:
            if live_plot and fig is not None:
                plt.ioff()
                if plt.fignum_exists(fig.number):
                    if train_batch_losses:
                        last_epoch = epoch if 'epoch' in locals() else epochs - 1
                        _update_live_chart(last_epoch, train_batch_losses[-1])
                        ax.set_title(
                            f"Batch Training Loss - Completed {len(train_batch_losses)} Batches "
                            f"(Final Loss: {train_batch_losses[-1]:.4f})"
                        )
                    if is_notebook and display_handle is not None:
                        display_handle.update(fig)
                        plt.close(fig)
                    else:
                        try:
                            plt.draw()
                            plt.show()
                        except Exception:
                            pass

        return train_batch_losses


class TransformerEvaluator:
    """Evaluate a transformer against an ordered item catalog."""

    def __init__(
        self,
        model: FurnitureEmbeddingPredictor,
        item_descriptions: list[str] = None,
        item_ids: list[int] = None,
        item_embeddings: torch.Tensor = None,
        return_strings = True
    ):
        self.model = model.to("cpu")
        self.model.eval()
        self.item_ids = item_ids
        self.item_descs = item_descriptions
        
        if item_embeddings is not None:
            if item_descriptions is None:
                raise ValueError("item_descriptions are required alongside item_embeddings")
            if item_embeddings.ndim != 2 or item_embeddings.shape[0] != len(item_descriptions):
                raise ValueError(
                    "item_embeddings must have one 2-D row for every item description"
                )
            if item_embeddings.shape[1] != model.embed_dim:
                raise ValueError(
                    f"item_embeddings must have dimension {model.embed_dim}, "
                    f"got {item_embeddings.shape[1]}"
                )
        self.return_strings = return_strings
        self.item_embeddings = item_embeddings.cpu() if item_embeddings is not None else None
        self.sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")

    def predict(self, room: list[str], k: int = 1) -> list:
        self.ensure_item_embeddings()
        
        if not room:
            raise ValueError("room must contain at least one item description")
        if self.item_descs is None:
            raise ValueError("item_descriptions are required to return ranked item strings")
        if k < 1:
            raise ValueError("k must be at least 1")
        
        with torch.no_grad():
            embedding: torch.Tensor = self.sbert.encode(
                room,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                ) # will have dim (n, embed_dim)
            embedding_cpu = embedding.unsqueeze(dim=0).cpu()
            #print(f"room embedding shape: {embedding.shape}, mask shape: {self.model.mask_token.shape}")

            embedding_masked = torch.cat([embedding_cpu, self.model.mask_token], dim = 1)
            valid_mask = torch.ones((1, embedding_masked.shape[1])).bool() # (1, n)
            prediction = self.model.forward(embedding_masked, valid_mask, mask_index=-1) # (1, embed_dim)
            sims = self.sbert.similarity(prediction, self.item_embeddings)[0]
            rankings = torch.argsort(sims, descending=True)
            ranked_strings = []
            ranked_ids = []
            present_items = set(room)
            
            for ranking in rankings:
                item = self.item_descs[int(ranking)]
                item_id = self.item_ids[int(ranking)]
                if item not in present_items:
                    ranked_strings.append(item)
                    ranked_ids.append(item_id)
                if len(ranked_strings) == k:
                    break
            
            if self.return_strings:
                return ranked_strings
            return ranked_ids

    def ensure_item_embeddings(self):
        if self.item_embeddings is not None:
            return
        if self.item_descs is None:
            raise ValueError("item_descriptions are required to build the candidate embedding matrix")

        self.item_embeddings = self.sbert.encode(
            self.item_descs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            )
        


if __name__ == '__main__':
    model = FurnitureEmbeddingPredictor()
    evaluator = TransformerEvaluator(model, item_descriptions=["chair", "wierd snake", "bedside table", "owen", "blender", "swimingpool"])

    prediction = evaluator.predict(["bed", "table"], k=1)
    print(prediction)



"""
x
shape = [B,L,384]

valid_mask
shape = [B,L]

mask_idx
shape = [B]

"""
