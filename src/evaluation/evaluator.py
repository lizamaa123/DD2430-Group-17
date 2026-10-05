import random
from src.evaluation.metrics import recall_at_k, ndcg_at_k

class LeaveOneOutEvaluator:
    def evaluate(self, model, test_rooms, k):
        random.seed(42)
        
        total_recall = 0.0
        total_ndcg = 0.0
        valid_rooms = 0

        for room in test_rooms:
            # skip if we only have one item in the room, cannot make predictions
            if len(room) < 2:
                continue 

            # room copy so we dont interfere with real data
            room_copy = room.copy()

            random_index = random.randint(0, len(room_copy) - 1)

            # target item will be randomized from copied room
            target_item = room_copy.pop(random_index)
            partial_room = room_copy # rest of the item inside the roomn = partial room

            # evaluator doesnt care WHAT the model is just that it has a predict method
            predictions = model.predict(partial_room, k=k)

            total_recall += recall_at_k(predictions, target_item, k)
            total_ndcg += ndcg_at_k(predictions, target_item, k)
            valid_rooms += 1

        # return the average recal and ndcg scores
        return total_recall / valid_rooms, total_ndcg / valid_rooms