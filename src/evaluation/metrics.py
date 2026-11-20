import math
import torch
import torch.nn.functional as F

def recall_at_k(predictions, target_item, k=10):
    # Recall@k returns 1.0 if the target item is in the top-k guesses
    # and returns 0.0 o.w
    # this metric does not consider the ranking position of the predicted target item, only that it is in our top-k guesses

    top_k_guesses = predictions[:k]
    if target_item in top_k_guesses:
        return 1.0
    else:
        return 0.0

def ndcg_at_k(predictions, target_item, k=10):
    # nDCG@k returns 1/log_2(i+1) where i=ranking position inside the top-k guesses (i≤k)
    # and returns 0 o.w
    top_k_guesses = predictions[:k]
    
    if target_item in top_k_guesses:
        # add 1 because lists start counting at index 0
        rank = top_k_guesses.index(target_item) + 1
        
        score = 1.0 / math.log2(rank + 1)
        return score
    # if target item not in top-k guesses
    else:
        return 0.0


def category_recall_at_k(predictions, target_item, category_mapping: dict, k=10):
    """Returns 1.0 if any of top-k predictions belongs to the same category as target_item, else 0.0."""
    target_cat = category_mapping.get(target_item)
    if target_cat is None:
        return 0.0

    top_k_guesses = predictions[:k]
    for item in top_k_guesses:
        if category_mapping.get(item) == target_cat:
            return 1.0
    return 0.0


def category_ndcg_at_k(predictions, target_item, category_mapping: dict, k=10):
    """Discounted ranking position of the first prediction matching target_item's category."""
    target_cat = category_mapping.get(target_item)
    if target_cat is None:
        return 0.0

    top_k_guesses = predictions[:k]
    for i, item in enumerate(top_k_guesses):
        if category_mapping.get(item) == target_cat:
            rank = i + 1
            return 1.0 / math.log2(rank + 1)
    return 0.0


def semantic_similarity_at_k(predictions, target_item, embedding_mapping: dict, k=10):
    """Maximum cosine similarity between target_item embedding and top-k predicted item embeddings."""
    target_emb = embedding_mapping.get(target_item)
    if target_emb is None:
        return 0.0

    top_k_guesses = predictions[:k]
    pred_embs = [embedding_mapping[item] for item in top_k_guesses if item in embedding_mapping]
    if not pred_embs:
        return 0.0

    if not isinstance(target_emb, torch.Tensor):
        target_emb = torch.tensor(target_emb, dtype=torch.float32)

    target_norm = F.normalize(target_emb.view(1, -1).float(), p=2, dim=-1)
    stacked_preds = torch.stack([
        e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)
        for e in pred_embs
    ]).float()
    preds_norm = F.normalize(stacked_preds, p=2, dim=-1)

    cos_sims = torch.matmul(target_norm, preds_norm.T).squeeze(0)
    return float(cos_sims.max().item())


def semantic_recall_at_k(
    predictions, target_item, embedding_mapping: dict, k=10, threshold=0.85
):
    """Returns 1.0 if target_item is in top-k or any top-k item has cosine similarity >= threshold."""
    if recall_at_k(predictions, target_item, k) == 1.0:
        return 1.0
    max_sim = semantic_similarity_at_k(predictions, target_item, embedding_mapping, k)
    return 1.0 if max_sim >= threshold else 0.0
