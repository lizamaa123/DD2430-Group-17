import random
from src.evaluation.metrics import recall_at_k, ndcg_at_k

class LeaveOneOutEvaluator:
    def evaluate(self, model, test_rooms, k):
        random.seed(42)
        
        total_recall = 0.0
        total_ndcg = 0.0
        valid_rooms = 0

        for room in test_rooms:
            # room copy so we dont interfere with real data
            # also considers room label if room is a dict
            if isinstance(room, dict):
                room_type = room["room_type"]
                room_copy = room["item_ids"].copy()
            else:
                room_type = None
                room_copy = room.copy()

            # skip if we only have one item in the room, cannot make predictions
            if len(room_copy) < 2:
                continue 

            random_index = random.randint(0, len(room_copy) - 1)

            # target item will be randomized from copied room
            target_item = room_copy.pop(random_index)
            partial_room = room_copy # rest of the item inside the roomn = partial room

            # evaluator doesnt care WHAT the model is just that it has a predict method
            if room_type is not None:
                predictions = model.predict(partial_room, k=k, room_type=room_type)
            else:
                predictions = model.predict(partial_room, k=k)

            total_recall += recall_at_k(predictions, target_item, k)
            total_ndcg += ndcg_at_k(predictions, target_item, k)
            valid_rooms += 1

        # return the average recal and ndcg scores
        return total_recall / valid_rooms, total_ndcg / valid_rooms

    def evaluate_per_category(self, model, test_rooms, item_to_category, k_values=(5, 10, 20)):
        """Same Leave-One-Out protocol as evaluate(), but results are grouped by the
        category of the hidden target item (category-level error analysis).

        Uses the same seed, so the hidden items are identical to evaluate().
        The model predicts max(k_values) items once, and Recall/NDCG for smaller k
        are computed from the top of that same list.

        Returns:
            dict: {category: {"n_rooms": int, "recall@k": float, "ndcg@k": float, ...}}
                  plus an "All" row with the overall averages.
        """
        random.seed(42)
        max_k = max(k_values)

        # running sums per category
        stats = {}

        for room in test_rooms:
            if isinstance(room, dict):
                room_type = room["room_type"]
                room_copy = room["item_ids"].copy()
            else:
                room_type = None
                room_copy = room.copy()

            if len(room_copy) < 2:
                continue

            random_index = random.randint(0, len(room_copy) - 1)
            target_item = room_copy.pop(random_index)
            partial_room = room_copy

            if room_type is not None:
                predictions = model.predict(partial_room, k=max_k, room_type=room_type)
            else:
                predictions = model.predict(partial_room, k=max_k)

            category = item_to_category.get(target_item, "Unknown")

            # add the result to both the category row and the overall row
            for key in (category, "All"):
                if key not in stats:
                    stats[key] = {"n_rooms": 0}
                    for k in k_values:
                        stats[key][f"recall@{k}"] = 0.0
                        stats[key][f"ndcg@{k}"] = 0.0
                stats[key]["n_rooms"] += 1
                for k in k_values:
                    stats[key][f"recall@{k}"] += recall_at_k(predictions, target_item, k)
                    stats[key][f"ndcg@{k}"] += ndcg_at_k(predictions, target_item, k)

        # turn sums into averages
        for key, row in stats.items():
            for metric in row:
                if metric != "n_rooms":
                    row[metric] /= row["n_rooms"]

        return stats
