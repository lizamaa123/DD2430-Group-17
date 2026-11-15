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

    def __init__(
        self,
        embed_dim=384,
        hid_dim=256,
        nhead=4,
        num_layers=3,
        num_candidates=3,
        device=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_candidates = num_candidates
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Loaded instance on device: {self.device} with {self.num_candidates} candidate head(s)")
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

        # Multi-candidate projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, hid_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(hid_dim, num_candidates * embed_dim, device=self.device),
        )

        # Certainty / confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, max(hid_dim // 2, num_candidates), device=self.device),
            nn.ReLU(),
            nn.Linear(max(hid_dim // 2, num_candidates), num_candidates, device=self.device),
        )

    def forward(self, x, valid_mask, mask_index, return_confidence=False):
        """Does inference for a batch of room sets with already embedded furniture/label vectors.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            valid_mask (torch.Tensor): Bool tensor of shape (batch_size, seq_len), True for real items.
            mask_index (int or torch.Tensor): The index of the furniture item to be masked in each sample.
            return_confidence (bool): If True, returns (preds, conf_logits).

        Returns:
            torch.Tensor or tuple:
                - If return_confidence is True: (preds, conf_logits) with preds shape (B, M, D)
                  and conf_logits shape (B, M).
                - If return_confidence is False and num_candidates == 1: preds shape (B, D).
                - If return_confidence is False and num_candidates > 1: preds shape (B, M, D).
        """

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
        hidden = out[batch_indices, mask_index, :]

        raw_preds = self.proj_head(hidden)
        preds = raw_preds.view(batch_size, self.num_candidates, self.embed_dim)
        preds = F.normalize(preds, p=2, dim=-1)

        conf_logits = self.conf_head(hidden)

        if return_confidence:
            return preds, conf_logits

        if self.num_candidates == 1:
            return preds.squeeze(1)

        return preds
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        labeled_data: bool = True,
        epochs=20,
        debug_prints=False,
        lr=2e-5,
        live_plot=True,
        plot_every=1,
        loss_type="infonce",
        temperature=0.07,
        conf_weight=0.2,
    ):
        """Trains the transformer using in-batch contrastive / InfoNCE loss or pairwise cosine similarity,
        with Winner-Take-All multi-head training and confidence prediction.

        Args:
            train_loader: DataLoader yielding batches of dicts with 'x' and 'valid_mask'.
            val_loader: Optional validation DataLoader.
            train_with_labels: Decides whether to insert the room label as the first item in the rooms for training transformer.
            epochs (int): Number of training epochs.
            debug_prints (bool): If True, prints batch-level losses.
            lr (float): Learning rate for AdamW optimizer.
            live_plot (bool): If True, displays a live updating matplotlib chart of batch loss during training.
            plot_every (int): Frequency (in batches) at which the live chart updates.
            loss_type (str): Loss function to use: 'infonce' (in-batch contrastive, default) or 'cosine'.
            temperature (float): Temperature parameter for InfoNCE scaling (default: 0.07).
            conf_weight (float): Weight multiplier for the confidence head cross-entropy loss (default: 0.2).
        """

        print(f"Training with device: {self.device} (loss: {loss_type}, temp: {temperature}, heads: {self.num_candidates})")
        self.to(self.device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        cos_loss = torch.nn.CosineEmbeddingLoss() if loss_type == "cosine" else None

        # logging
        train_batch_losses = []

        loss_label = "InfoNCE Loss" if loss_type == "infonce" else "Cosine Loss"

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
            ax.set_title(f"Training Loss ({loss_label})")
            ax.set_xlabel("Batch")
            ax.set_ylabel(loss_label)
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
                f"Batch Training Loss ({loss_label}) - Epoch {epoch_idx + 1}/{epochs} | "
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
                    mask_probs = valid_mask.float()
                    if labeled_data: 
                        mask_probs[:, 0] = 0.0

                    mask_idx = torch.multinomial(mask_probs,1).squeeze(1)

                    # Save the true furniture embedding before masking.
                    true = x[batch_indices, mask_idx,:]

                    optimizer.zero_grad()

                    # Forward pass returning candidate predictions and confidence logits
                    preds, conf_logits = self.forward(x, valid_mask, mask_idx, return_confidence=True)
                    true_norm = F.normalize(true, p=2, dim=-1)

                    # Winner-Take-All: determine which candidate head is closest to the ground truth
                    head_sims = torch.sum(preds * true_norm.unsqueeze(1), dim=-1)  # shape: [batch_size, num_candidates]
                    winner_idx = torch.argmax(head_sims, dim=-1)                    # shape: [batch_size]
                    winning_preds = preds[batch_indices, winner_idx, :]            # shape: [batch_size, embed_dim]

                    if loss_type == "infonce":
                        # In-batch contrastive / InfoNCE loss on the winning candidate
                        sim_matrix = torch.matmul(winning_preds, true_norm.T) / temperature
                        targets = torch.arange(batch_size, device=self.device)
                        embed_loss = F.cross_entropy(sim_matrix, targets)
                    elif loss_type == "cosine":
                        embed_loss = (1.0 - head_sims[batch_indices, winner_idx]).mean()
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'infonce' or 'cosine'.")

                    # Confidence loss: train the certainty head to predict which candidate won
                    if self.num_candidates > 1:
                        conf_loss = F.cross_entropy(conf_logits, winner_idx)
                        loss = embed_loss + conf_weight * conf_loss
                    else:
                        loss = embed_loss

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
        aggregation = "confidence_weighted_max",
    ):
        self.model = model.to("cpu")
        self.model.eval()
        self.item_descs = item_descriptions
        self.aggregation = aggregation
    
        self.item_embeddings = None
        self.sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")
        self.ensure_item_embeddings()
        self.desc_to_emb = dict(zip(self.item_descs, self.item_embeddings))




    def predict(self, room: list[str], k: int = 1) -> list:
        if not room:
            raise ValueError("room must contain at least one item description")
        if self.item_descs is None:
            raise ValueError("item_descriptions are required to return ranked item strings")
        if k < 1:
            raise ValueError("k must be at least 1")
        
        with torch.no_grad():
            for item in room:
                if item not in self.desc_to_emb:
                    self.desc_to_emb[item] = self.sbert.encode(
                                        item,
                                        convert_to_tensor=True,
                                        normalize_embeddings=True,
                                        show_progress_bar=False,
                                        )
                    print(f"Did not find embedding for: {item}")

            embedding = torch.stack([self.desc_to_emb[item] for item in room], dim=0).unsqueeze(dim=0).cpu()
            embedding_masked = torch.cat([embedding, self.model.mask_token], dim = 1)
            valid_mask = torch.ones((1, embedding_masked.shape[1])).bool() # (1, n)
            preds, conf_logits = self.model.forward(embedding_masked, valid_mask, mask_index=-1, return_confidence=True)
            
            probs = F.softmax(conf_logits, dim=-1)[0]
            catalog = F.normalize(self.item_embeddings, p=2, dim=-1)
            head_sims = torch.matmul(preds[0], catalog.T)

            if self.aggregation == "weighted":
                sims = torch.matmul(probs, head_sims)
            elif self.aggregation == "confidence_weighted_max":
                sims = torch.max(head_sims * probs.unsqueeze(1), dim=0).values
            else:  # "max" (default)
                sims = torch.max(head_sims, dim=0).values

            rankings = torch.argsort(sims, descending=True)
            ranked_strings = []
            present_items = set(room)
            
            for ranking in rankings:
                item = self.item_descs[int(ranking)]
                #item_id = self.item_ids[int(ranking)]
                if item not in present_items:
                    ranked_strings.append(item)
                    #ranked_ids.append(item_id)
                if len(ranked_strings) == k:
                    break

            return ranked_strings
  

    def predict_choices(self, room: list[str], k_per_head: int = 1) -> list[dict]:
        """Returns predictions broken down by individual candidate heads with their confidence probabilities."""
        if not room:
            raise ValueError("room must contain at least one item description")

        with torch.no_grad():
            embedding = self.sbert.encode(
                room,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).unsqueeze(dim=0).cpu()

            embedding_masked = torch.cat([embedding, self.model.mask_token], dim=1)
            valid_mask = torch.ones((1, embedding_masked.shape[1])).bool()

            preds, conf_logits = self.model.forward(
                embedding_masked, valid_mask, mask_index=-1, return_confidence=True
            )
            probs = F.softmax(conf_logits, dim=-1)[0].tolist()
            catalog = F.normalize(self.item_embeddings, p=2, dim=-1)
            head_sims = torch.matmul(preds[0], catalog.T)

            present_items = set(room)
            choices = []
            for m in range(self.model.num_candidates):
                rankings = torch.argsort(head_sims[m], descending=True)
                items = []
                for r in rankings:
                    idx = int(r)
                    item = self.item_descs[idx] if self.return_strings else self.item_ids[idx]
                    if self.item_descs[idx] not in present_items:
                        items.append(item)
                    if len(items) == k_per_head:
                        break
                choices.append({
                    "head": m,
                    "confidence": float(probs[m]),
                    "predictions": items,
                })

            choices.sort(key=lambda x: x["confidence"], reverse=True)
            return choices

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
