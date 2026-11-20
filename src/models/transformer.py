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
        hid_dim=512,
        nhead=4,
        num_layers=3,
        num_candidates=3,
        dropout = 0.05,
        device=None
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
            device=self.device,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
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
        train_batch_accs = []
        train_head_wins = []
        val_losses = []
        val_accs = []
        val_batch_indices = []

        loss_label = "InfoNCE Loss" if loss_type == "infonce" else "Cosine Loss"

        # Setup live plotting if requested
        is_notebook = False
        display_handle = None
        fig = None
        ax_loss = None
        ax_acc = None
        ax_heads = None
        loss_line = None
        smooth_loss_line = None
        val_loss_line = None
        acc_line = None
        smooth_acc_line = None
        val_acc_line = None
        head_colors = [plt.get_cmap("tab10")(i % 10) for i in range(self.num_candidates)]

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
            if self.num_candidates > 1:
                fig, (ax_loss, ax_acc, ax_heads) = plt.subplots(1, 3, figsize=(19, 5))
            else:
                fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss Subplot
            (loss_line,) = ax_loss.plot(
                [], [], label="Train Batch Loss", color="royalblue", alpha=0.35, linewidth=1
            )
            (smooth_loss_line,) = ax_loss.plot(
                [], [], label="Train Running Avg", color="royalblue", linewidth=1.8
            )
            (val_loss_line,) = ax_loss.plot(
                [], [], "o-", label="Val Loss", color="crimson", linewidth=2, markersize=5
            )
            ax_loss.set_title(f"Loss ({loss_label})")
            ax_loss.set_xlabel("Batch")
            ax_loss.set_ylabel(loss_label)
            ax_loss.grid(True, linestyle="--", alpha=0.6)
            ax_loss.legend(loc="upper right")

            # Accuracy Subplot
            (acc_line,) = ax_acc.plot(
                [], [], label="Train Batch Acc", color="mediumseagreen", alpha=0.35, linewidth=1
            )
            (smooth_acc_line,) = ax_acc.plot(
                [], [], label="Train Running Avg", color="mediumseagreen", linewidth=1.8
            )
            (val_acc_line,) = ax_acc.plot(
                [], [], "o-", label="Val Acc", color="darkorange", linewidth=2, markersize=5
            )
            ax_acc.set_title("Top-1 Accuracy")
            ax_acc.set_xlabel("Batch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_ylim(-0.02, 1.02)
            ax_acc.grid(True, linestyle="--", alpha=0.6)
            ax_acc.legend(loc="upper right")

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

        def _update_live_chart(epoch_idx: int, current_loss: float, current_acc: float):
            if fig is None or not plt.fignum_exists(fig.number):
                return
            n_batches = len(train_batch_losses)
            if n_batches == 0:
                return
            x_data = np.arange(1, n_batches + 1)

            # Update Loss curves
            loss_line.set_data(x_data, train_batch_losses)
            smooth_loss = calculate_running_average(train_batch_losses, window_size=50)
            smooth_loss_line.set_data(x_data, smooth_loss)
            if val_batch_indices and val_losses:
                val_loss_line.set_data(val_batch_indices, val_losses)

            ax_loss.relim()
            ax_loss.autoscale_view()
            val_loss_str = f" | Val Loss: {val_losses[-1]:.4f}" if val_losses else ""
            ax_loss.set_title(
                f"Loss ({loss_label}) - Epoch {epoch_idx + 1}/{epochs} | "
                f"Train: {current_loss:.4f}{val_loss_str}"
            )

            # Update Accuracy curves
            acc_line.set_data(x_data, train_batch_accs)
            smooth_acc = calculate_running_average(train_batch_accs, window_size=50)
            smooth_acc_line.set_data(x_data, smooth_acc)
            if val_batch_indices and val_accs:
                val_acc_line.set_data(val_batch_indices, val_accs)

            ax_acc.relim()
            ax_acc.autoscale_view()
            ax_acc.set_ylim(-0.02, 1.02)
            val_acc_str = f" | Val Acc: {val_accs[-1]:.4f}" if val_accs else ""
            ax_acc.set_title(
                f"Top-1 Accuracy - Epoch {epoch_idx + 1}/{epochs} | "
                f"Train: {current_acc:.4f}{val_acc_str}"
            )

            # Update Winning Head distribution (100% stacked area chart)
            if ax_heads is not None and train_head_wins:
                wins_arr = np.array(train_head_wins)
                smoothed_pcts = [
                    calculate_running_average(wins_arr[:, h], window_size=50) * 100.0
                    for h in range(self.num_candidates)
                ]
                latest_pcts = [float(smoothed_pcts[h][-1]) for h in range(self.num_candidates)]
                ax_heads.clear()
                ax_heads.stackplot(
                    x_data,
                    *smoothed_pcts,
                    labels=[f"H{h}: {latest_pcts[h]:.1f}%" for h in range(self.num_candidates)],
                    colors=head_colors,
                    alpha=0.85,
                )
                ax_heads.set_ylim(0, 100)
                ax_heads.set_xlim(1, max(n_batches, 2))
                ax_heads.set_xlabel("Batch")
                ax_heads.set_ylabel("Win Share (%)")
                ax_heads.grid(True, linestyle="--", alpha=0.4)
                legend_cols = 1 if self.num_candidates <= 7 else (2 if self.num_candidates <= 14 else 3)
                legend_fs = "small" if self.num_candidates <= 7 else "x-small"
                ax_heads.legend(loc="upper right", ncol=legend_cols, fontsize=legend_fs)
                if self.num_candidates <= 4:
                    latest_shares = " | ".join(
                        f"H{h}: {latest_pcts[h]:.1f}%" for h in range(self.num_candidates)
                    )
                    ax_heads.set_title(f"Winning Head % - {latest_shares}")
                else:
                    active_count = sum(1 for p in latest_pcts if p > 1.0)
                    top_head = int(np.argmax(latest_pcts))
                    ax_heads.set_title(
                        f"Winning Head % | Active (>1%): {active_count}/{self.num_candidates} "
                        f"| Top: H{top_head} ({latest_pcts[top_head]:.1f}%)"
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
                epoch_acc = 0

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
                        batch_acc = (sim_matrix.argmax(dim=-1) == targets).float().mean().item()
                    elif loss_type == "cosine":
                        embed_loss = (1.0 - head_sims[batch_indices, winner_idx]).mean()
                        sim_matrix = torch.matmul(winning_preds, true_norm.T)
                        batch_acc = (sim_matrix.argmax(dim=-1) == torch.arange(batch_size, device=self.device)).float().mean().item()
                        #batch_acc = (head_sims[batch_indices, winner_idx] > 0.5).float().mean().item()
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
                    epoch_acc += batch_acc
                    train_batch_losses.append(batch_loss_val)
                    train_batch_accs.append(batch_acc)
                    head_fracs = (
                        torch.bincount(winner_idx, minlength=self.num_candidates).float() / batch_size
                    )
                    train_head_wins.append(head_fracs.detach().cpu().numpy())

                    if live_plot and (len(train_batch_losses) % plot_every == 0):
                        _update_live_chart(epoch, batch_loss_val, batch_acc)

                    if debug_prints:
                        print(f"batch loss: {batch_loss_val}")

                epoch_loss /= len(train_loader)
                epoch_acc /= len(train_loader)

                # Validation pass
                val_loss = None
                val_acc = None
                if val_loader is not None:
                    self.eval()
                    total_val_loss = 0.0
                    total_val_acc = 0.0
                    val_batches = 0
                    val_rng = torch.Generator(device=self.device).manual_seed(42 + epoch)

                    with torch.no_grad():
                        for val_batch in val_loader:
                            x_val = val_batch["x"].to(self.device)
                            valid_mask_val = val_batch["valid_mask"].to(self.device)
                            b_size_val = x_val.size(0)
                            if b_size_val < 2:
                                continue

                            b_indices_val = torch.arange(b_size_val, device=self.device)
                            mask_probs_val = valid_mask_val.float()
                            if labeled_data:
                                mask_probs_val[:, 0] = 0.0

                            mask_idx_val = torch.multinomial(mask_probs_val, 1, generator=val_rng).squeeze(1)
                            true_val = x_val[b_indices_val, mask_idx_val, :]
                            true_val_norm = F.normalize(true_val, p=2, dim=-1)

                            preds_val, conf_logits_val = self.forward(
                                x_val, valid_mask_val, mask_idx_val, return_confidence=True
                            )
                            head_sims_val = torch.sum(preds_val * true_val_norm.unsqueeze(1), dim=-1)
                            winner_idx_val = torch.argmax(head_sims_val, dim=-1)
                            winning_preds_val = preds_val[b_indices_val, winner_idx_val, :]

                            targets_val = torch.arange(b_size_val, device=self.device)
                            if loss_type == "infonce":
                                sim_mat_val = torch.matmul(winning_preds_val, true_val_norm.T) / temperature
                                e_loss_val = F.cross_entropy(sim_mat_val, targets_val)
                                b_acc_val = (sim_mat_val.argmax(dim=-1) == targets_val).float().mean().item()
                            else:
                                e_loss_val = (1.0 - head_sims_val[b_indices_val, winner_idx_val]).mean()
                                #b_acc_val = (head_sims_val[b_indices_val, winner_idx_val] > 0.5).float().mean().item()
                                sim_matrix = torch.matmul(winning_preds, true_norm.T)
                                b_acc_val = (sim_matrix.argmax(dim=-1) == torch.arange(batch_size, device=self.device)).float().mean().item()

                            if self.num_candidates > 1:
                                c_loss_val = F.cross_entropy(conf_logits_val, winner_idx_val)
                                b_loss_val = e_loss_val + conf_weight * c_loss_val
                            else:
                                b_loss_val = e_loss_val

                            total_val_loss += b_loss_val.item()
                            total_val_acc += b_acc_val
                            val_batches += 1

                    if val_batches > 0:
                        val_loss = total_val_loss / val_batches
                        val_acc = total_val_acc / val_batches
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                        val_batch_indices.append(len(train_batch_losses))

                    self.train()

                if live_plot:
                    _update_live_chart(epoch, batch_loss_val, batch_acc)

                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loss is not None else ""
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}"
                    f"{val_info}"
                )

        finally:
            if live_plot and fig is not None:
                plt.ioff()
                if plt.fignum_exists(fig.number):
                    if train_batch_losses:
                        last_epoch = epoch if 'epoch' in locals() else epochs - 1
                        _update_live_chart(last_epoch, train_batch_losses[-1], train_batch_accs[-1])
                        val_loss_str = f" | Val Loss: {val_losses[-1]:.4f}" if val_losses else ""
                        val_acc_str = f" | Val Acc: {val_accs[-1]:.4f}" if val_accs else ""
                        ax_loss.set_title(
                            f"Loss - {len(train_batch_losses)} Batches | Final Train: {train_batch_losses[-1]:.4f}{val_loss_str}"
                        )
                        ax_acc.set_title(
                            f"Accuracy - Final Train: {train_batch_accs[-1]:.4f}{val_acc_str}"
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
    """Evaluate a transformer against a catalog using precomputed embeddings."""

    def __init__(
        self,
        model: FurnitureEmbeddingPredictor,
        item_embeddings: dict[int, torch.Tensor] | torch.Tensor,
        item_ids: list[int] = None,
        item_descriptions: list[str] = None,
        return_strings: bool = False,
        aggregation: str = "confidence_weighted_max",
        filter_present: bool = True,
        train_rooms: list[list] = None,
        item_counts: dict = None,
        pop_weight: float = 0.0,
        room_type_embeddings: dict[str, torch.Tensor] = None,
    ):
        self.model = model.to("cpu")
        self.model.eval()
        self.aggregation = aggregation
        self.filter_present = filter_present
        self.item_descriptions = item_descriptions
        self.return_strings = return_strings
        self.pop_weight = pop_weight
        self.room_type_embeddings = (
            {k: v.cpu() for k, v in room_type_embeddings.items()}
            if room_type_embeddings is not None
            else None
        )

        # Case 1: item_embeddings is a dictionary {item_id: Tensor(embed_dim)}
        if isinstance(item_embeddings, dict):
            self.id_to_emb = {k: v.cpu() for k, v in item_embeddings.items()}
            self.item_ids = list(self.id_to_emb.keys()) if item_ids is None else list(item_ids)
            self.catalog_matrix = torch.stack([self.id_to_emb[iid] for iid in self.item_ids])

        # Case 2: item_embeddings is a 2D Tensor of shape (N, embed_dim)
        elif isinstance(item_embeddings, torch.Tensor):
            if item_ids is None:
                raise ValueError("item_ids must be provided when item_embeddings is a 2D Tensor")
            self.item_ids = list(item_ids)
            self.catalog_matrix = item_embeddings.cpu()
            self.id_to_emb = {iid: emb for iid, emb in zip(self.item_ids, self.catalog_matrix)}
        else:
            raise TypeError("item_embeddings must be either a dict or a torch.Tensor")

        if self.item_descriptions is not None:
            self.desc_to_emb = {desc: emb for desc, emb in zip(self.item_descriptions, self.catalog_matrix)}

        # Pre-normalize catalog matrix once for fast cosine similarity dot products
        self.catalog_norm = F.normalize(self.catalog_matrix, p=2, dim=-1)

        # Optional normalized log-popularity tie-breaker aligned with self.item_ids
        self.pop_bonus = None
        if pop_weight > 0.0 and (item_counts is not None or train_rooms is not None):
            if item_counts is None:
                from collections import Counter
                item_counts = Counter(iid for room in train_rooms for iid in room)
            counts = torch.tensor(
                [float(item_counts.get(iid, 0)) for iid in self.item_ids],
                dtype=torch.float32,
            )
            log_counts = torch.log1p(counts)
            max_log = log_counts.max()
            if max_log > 0:
                self.pop_bonus = log_counts / max_log

    def _get_item_emb(self, item):
        if item in self.id_to_emb:
            return self.id_to_emb[item]
        if self.room_type_embeddings is not None:
            if item in self.room_type_embeddings:
                return self.room_type_embeddings[item]
            if isinstance(item, str) and item.replace("_", " ") in self.room_type_embeddings:
                return self.room_type_embeddings[item.replace("_", " ")]
        if hasattr(self, 'desc_to_emb') and item in self.desc_to_emb:
            return self.desc_to_emb[item]
        raise KeyError(f"Item '{item}' not found in catalog embeddings.")

    def predict(self, room: list, k: int = 1) -> list:
        if not room:
            raise ValueError("room must contain at least one item")
        if k < 1:
            raise ValueError("k must be at least 1")

        with torch.no_grad():
            # 1. Look up embeddings for the input room items (shape: 1, seq_len, embed_dim)
            embedding = torch.stack([self._get_item_emb(item) for item in room], dim=0).unsqueeze(0).cpu()

            # 2. Append mask token
            mask_token = self.model.mask_token.to(embedding.device)
            embedding_masked = torch.cat([embedding, mask_token], dim=1)
            valid_mask = torch.ones((1, embedding_masked.shape[1]), dtype=torch.bool, device=embedding.device)

            # 3. Predict candidate vectors & confidence logits
            preds, conf_logits = self.model.forward(
                embedding_masked, valid_mask, mask_index=-1, return_confidence=True
            )

            # 4. Cosine similarity against normalized catalog
            probs = F.softmax(conf_logits, dim=-1)[0]  # shape: (num_candidates,)
            head_sims = torch.matmul(preds[0], self.catalog_norm.T)  # shape: (num_candidates, N)
            if self.pop_bonus is not None and self.pop_weight > 0.0:
                head_sims = head_sims + self.pop_weight * self.pop_bonus.unsqueeze(0)

            # 5. Aggregate candidate heads
            if self.aggregation in ("interleaved", "sampled_interleaved"):
                ranked_results = []
                seen = set()
                present_items = set(room) if self.filter_present else set()
                num_heads = head_sims.shape[0]
                max_fetch = min(head_sims.shape[1], k + len(present_items) + 5)
                head_top_indices = {
                    h: torch.topk(head_sims[h], k=max_fetch).indices.tolist()
                    for h in range(num_heads)
                }
                pointers = {h: 0 for h in range(num_heads)}

                def _pop_next_from_head(h: int) -> bool:
                    indices_h = head_top_indices[h]
                    while pointers[h] < len(indices_h):
                        idx = indices_h[pointers[h]]
                        pointers[h] += 1
                        item_val = (
                            self.item_descriptions[idx]
                            if (self.return_strings and self.item_descriptions is not None)
                            else self.item_ids[idx]
                        )
                        if item_val not in present_items and item_val not in seen:
                            seen.add(item_val)
                            ranked_results.append(item_val)
                            return True
                    return False

                if self.aggregation == "interleaved":
                    head_order = torch.argsort(probs, descending=True).tolist()
                    while len(ranked_results) < k:
                        added_any = False
                        for h in head_order:
                            if _pop_next_from_head(h):
                                added_any = True
                            if len(ranked_results) == k:
                                break
                        if not added_any:
                            break
                else:  # "sampled_interleaved"
                    cat_dist = torch.distributions.Categorical(probs=probs)
                    exhausted = set()
                    while len(ranked_results) < k and len(exhausted) < num_heads:
                        h = int(cat_dist.sample())
                        if h in exhausted:
                            continue
                        if not _pop_next_from_head(h):
                            exhausted.add(h)

                return ranked_results

            if self.aggregation == "weighted":
                sims = torch.matmul(probs, head_sims)
            elif self.aggregation == "confidence_weighted_max":
                sims = torch.max(head_sims * probs.unsqueeze(1), dim=0).values
            else:  # "max"
                sims = torch.max(head_sims, dim=0).values

            # 6. Sort candidate indices descending
            rankings = torch.argsort(sims, descending=True)

            # 7. Map row indices back to item IDs or descriptions
            ranked_results = []
            seen = set()
            present_items = set(room) if self.filter_present else set()

            for r in rankings:
                idx = int(r)
                item_val = self.item_descriptions[idx] if (self.return_strings and self.item_descriptions is not None) else self.item_ids[idx]

                if item_val not in present_items and item_val not in seen:
                    seen.add(item_val)
                    ranked_results.append(item_val)

                if len(ranked_results) == k:
                    break

            return ranked_results

    def predict_choices(self, room: list, k_per_head: int = 1) -> list[dict]:
        """Returns predictions broken down by individual candidate head."""
        if not room:
            raise ValueError("room must contain at least one item")

        with torch.no_grad():
            embedding = torch.stack([self._get_item_emb(item_id) for item_id in room], dim=0).unsqueeze(0).cpu()
            mask_token = self.model.mask_token.to(embedding.device)
            embedding_masked = torch.cat([embedding, mask_token], dim=1)
            valid_mask = torch.ones((1, embedding_masked.shape[1]), dtype=torch.bool, device=embedding.device)

            preds, conf_logits = self.model.forward(
                embedding_masked, valid_mask, mask_index=-1, return_confidence=True
            )
            probs = F.softmax(conf_logits, dim=-1)[0].tolist()
            head_sims = torch.matmul(preds[0], self.catalog_norm.T)

            present_items = set(room) if self.filter_present else set()
            choices = []
            for m in range(self.model.num_candidates):
                rankings = torch.argsort(head_sims[m], descending=True)
                items = []
                seen = set()
                for r in rankings:
                    idx = int(r)
                    iid = self.item_descriptions[idx] if (self.return_strings and self.item_descriptions is not None) else self.item_ids[idx]
                    if iid not in present_items and iid not in seen:
                        seen.add(iid)
                        items.append(iid)
                    if len(items) == k_per_head:
                        break
                choices.append({
                    "head": m,
                    "confidence": float(probs[m]),
                    "predictions": items,
                })

            choices.sort(key=lambda x: x["confidence"], reverse=True)
            return choices

"""
x
shape = [B,L,384]

valid_mask
shape = [B,L]

mask_idx
shape = [B]

"""
