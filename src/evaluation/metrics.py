import math

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