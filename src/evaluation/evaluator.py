import random
from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    category_recall_at_k,
    category_ndcg_at_k,
    semantic_similarity_at_k,
    semantic_recall_at_k,
)
from tqdm import tqdm


class LeaveOneOutEvaluator:
    def __init__(
        self,
        k: int = 10,
        category_mapping: dict = None,
        embedding_mapping: dict = None,
        similarity_threshold: float = 0.85,
    ):
        self.k = k
        self.category_mapping = category_mapping
        self.embedding_mapping = embedding_mapping
        self.similarity_threshold = similarity_threshold
        self.last_results = {}

    def evaluate(
        self,
        model,
        test_rooms,
        k: int = None,
        labeled: bool = False,
        category_mapping: dict = None,
        embedding_mapping: dict = None,
        similarity_threshold: float = None,
        return_dict: bool = False,
    ):
        eval_k = k if k is not None else self.k
        cat_map = category_mapping if category_mapping is not None else self.category_mapping
        emb_map = embedding_mapping if embedding_mapping is not None else self.embedding_mapping
        sim_thresh = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        random.seed(42)

        total_recall = 0.0
        total_ndcg = 0.0
        total_cat_recall = 0.0
        total_cat_ndcg = 0.0
        total_sim = 0.0
        total_sem_recall = 0.0
        valid_rooms = 0

        pbar = tqdm(range(len(test_rooms)))
        for i in pbar:
            room = test_rooms[i]
            # skip if we only have one item in the room, cannot make predictions
            if len(room) < 2:
                continue

            # room copy so we dont interfere with real data
            room_copy = room.copy()
            if labeled:
                random_index = random.randint(1, len(room_copy) - 1)
            else:
                random_index = random.randint(0, len(room_copy) - 1)

            # target item will be randomized from copied room
            target_item = room_copy.pop(random_index)
            partial_room = room_copy

            predictions = model.predict(partial_room, k=eval_k)

            # 1. Exact Item ID metrics
            total_recall += recall_at_k(predictions, target_item, eval_k)
            total_ndcg += ndcg_at_k(predictions, target_item, eval_k)

            # 2. Category-level metrics
            if cat_map is not None:
                total_cat_recall += category_recall_at_k(predictions, target_item, cat_map, eval_k)
                total_cat_ndcg += category_ndcg_at_k(predictions, target_item, cat_map, eval_k)

            # 3. Semantic similarity & semantic recall metrics
            if emb_map is not None:
                total_sim += semantic_similarity_at_k(predictions, target_item, emb_map, eval_k)
                total_sem_recall += semantic_recall_at_k(
                    predictions, target_item, emb_map, eval_k, threshold=sim_thresh
                )

            valid_rooms += 1

        if valid_rooms == 0:
            avg_recall = avg_ndcg = 0.0
            avg_cat_recall = avg_cat_ndcg = avg_sim = avg_sem_recall = 0.0
        else:
            avg_recall = total_recall / valid_rooms
            avg_ndcg = total_ndcg / valid_rooms
            avg_cat_recall = total_cat_recall / valid_rooms if cat_map is not None else None
            avg_cat_ndcg = total_cat_ndcg / valid_rooms if cat_map is not None else None
            avg_sim = total_sim / valid_rooms if emb_map is not None else None
            avg_sem_recall = total_sem_recall / valid_rooms if emb_map is not None else None

        self.last_results = {
            "exact_recall": avg_recall,
            "exact_ndcg": avg_ndcg,
        }
        if cat_map is not None:
            self.last_results["category_recall"] = avg_cat_recall
            self.last_results["category_ndcg"] = avg_cat_ndcg
        if emb_map is not None:
            self.last_results["semantic_similarity"] = avg_sim
            self.last_results["semantic_recall"] = avg_sem_recall

        if return_dict:
            return self.last_results

        return avg_recall, avg_ndcg

    def evaluate_all(
        self,
        model,
        test_rooms,
        k: int = None,
        labeled: bool = False,
        category_mapping: dict = None,
        embedding_mapping: dict = None,
        similarity_threshold: float = None,
    ):
        """Runs evaluation and returns a complete dictionary of all metrics."""
        return self.evaluate(
            model=model,
            test_rooms=test_rooms,
            k=k,
            labeled=labeled,
            category_mapping=category_mapping,
            embedding_mapping=embedding_mapping,
            similarity_threshold=similarity_threshold,
            return_dict=True,
        )
