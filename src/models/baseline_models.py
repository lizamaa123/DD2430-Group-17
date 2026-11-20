import numpy as np
from sentence_transformers import SentenceTransformer
# help taken from: https://sbert.net

class PopularityPrior():
    def __init__(self):
        # score final sorted list of popular items
        self.popularity_ranking = []
        # sorted popular items per room_type
        self.room_popularity = {}

    def fit(self, training_rooms):
        item_tally = {}
        room_tally = {}

        # loop through every room and then loop through all items
        for room in training_rooms:
            if isinstance(room, dict):
                room_type = room["room_type"].replace("_", " ") if room.get("room_type") else None
                items = room["item_ids"]
            else:
                room_type = None
                items = room

            for item in items:
                # if we have seen item before we add 1 to its tally
                item_tally[item] = item_tally.get(item, 0) + 1

                if room_type is not None:
                    if room_type not in room_tally:
                        room_tally[room_type] = {}
                    room_tally[room_type][item] = room_tally[room_type].get(item, 0) + 1

        sorted_items = sorted(item_tally.items(), key=lambda x: x[1], reverse=True)
        self.popularity_ranking = [item[0] for item in sorted_items]

        for r_type, tally in room_tally.items():
            sorted_r_items = sorted(tally.items(), key=lambda x: x[1], reverse=True)
            self.room_popularity[r_type] = [item[0] for item in sorted_r_items]

    def predict(self, partial_room, k=10, room_type=None):
        predicitions = []
        items_in_room = partial_room

        if room_type is None and partial_room and isinstance(partial_room[0], str):
            candidate_rt = partial_room[0].replace("_", " ")
            if candidate_rt in self.room_popularity:
                room_type = candidate_rt
                items_in_room = partial_room[1:]
        elif isinstance(room_type, str):
            room_type = room_type.replace("_", " ")

        ranking = self.room_popularity.get(room_type, self.popularity_ranking)

        # loop through first item in our "popular items list"
        for item in ranking:
            # only recommend popular item if its not already in the room
            if item not in items_in_room:
                predicitions.append(item)

            # stop guessing after k guesses 
            if len(predicitions) == k:
                break

        # fallback to global popularity if room_type ranking had fewer than k items
        if len(predicitions) < k:
            for item in self.popularity_ranking:
                if item not in items_in_room and item not in predicitions:
                    predicitions.append(item)
                if len(predicitions) == k:
                    break

        return predicitions
    
class ItemContentBaseline:
    def __init__(
        self,
        item_id,
        item_descriptions=None,
        return_descriptions=False,
        item_embeddings: dict = None,
        room_type_embeddings: dict = None,
    ):
        # Keep IDs and descriptions aligned with the embedding rows.
        self.item_id = np.array(item_id)
        self.item_descriptions = list(item_descriptions) if item_descriptions is not None else None
        self.return_descriptions = return_descriptions
        self._id_to_row = {item: row for row, item in enumerate(self.item_id)}
        self._description_to_row = (
            {description: row for row, description in enumerate(self.item_descriptions)}
            if self.item_descriptions is not None
            else {}
        )

        if item_embeddings is not None:
            self.model = None
            self.item_embedding = np.stack(
                [item_embeddings[int(iid)].detach().cpu().numpy() for iid in self.item_id]
            ).astype(np.float32)
        else:
            # loading pre-trained SBERT
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # create embeddings of item descriptions
            self.item_embedding = self.model.encode(self.item_descriptions)

        # Pre-normalize catalog embeddings for fast cosine similarity
        norms = np.linalg.norm(self.item_embedding, axis=1, keepdims=True)
        self._norm_item_embedding = self.item_embedding / np.maximum(norms, 1e-12)

        self.room_type_embeddings = (
            {
                k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                for k, v in room_type_embeddings.items()
            }
            if room_type_embeddings is not None
            else {}
        )

    def fit(self, training_rooms):
        # this baseline model doesn't need to learn from room groupings
        # so this method exist so the evaluation engine doesnt crash
        pass

    def predict(self, partial_room, k=10, room_type=None):
        lookup = (
            self._description_to_row
            if self.return_descriptions
            else self._id_to_row
        )

        items_in_room = partial_room
        if room_type is None and not self.return_descriptions and partial_room and isinstance(partial_room[0], str):
            candidate_rt = partial_room[0].replace("_", " ")
            if candidate_rt in self.room_type_embeddings:
                room_type = candidate_rt
                items_in_room = partial_room[1:]
        elif isinstance(room_type, str):
            room_type = room_type.replace("_", " ")

        present_item_rows = [lookup[item] for item in items_in_room if item in lookup]

        if not present_item_rows and (room_type is None or room_type not in self.room_type_embeddings):
            # if we do not have valid items we just return the first k catalog items
            if self.return_descriptions and self.item_descriptions is not None:
                return self.item_descriptions[:k]
            return self.item_id[:k].tolist()

        vecs = []
        if present_item_rows:
            vecs.append(self.item_embedding[present_item_rows])
        if room_type is not None and room_type in self.room_type_embeddings:
            vecs.append(self.room_type_embeddings[room_type].reshape(1, -1))

        present_item_embedding = np.vstack(vecs)

        # take the average of room embeddings to create a single "room profile" embedding
        room_embedding = np.mean(present_item_embedding, axis=0)
        room_norm = room_embedding / max(np.linalg.norm(room_embedding), 1e-12)

        # calculate cosine similarity between the room profile and every item
        similarities = self._norm_item_embedding @ room_norm

        # sort by highest similarity score
        sorted_indices = np.argsort(similarities)[::-1]

        # pick the top k items that are not already in the room = predictions
        predictions = []
        present_items = set(items_in_room)
        for row in sorted_indices:
            item = (
                self.item_descriptions[row]
                if self.return_descriptions
                else self.item_id[row]
            )
            if item not in present_items:
                predictions.append(item)
            if len(predictions) == k:
                break

        return predictions

class ItemItemCooccurrence:
    def __init__(self):
        # a nested dictionary to store pairwise counts
        self.co_counts = {}
        # backup list in case we cant find enough matches
        self.popular_backup = []
        # per-room-type co-occurrence counts and popular backups
        self.room_co_counts = {}
        self.room_popular_backup = {}

    def fit(self, training_rooms):
        item_tally = {}
        room_tally = {}
        
        for room in training_rooms:
            if isinstance(room, dict):
                room_type = room["room_type"].replace("_", " ") if room.get("room_type") else None
                items = room["item_ids"]
            else:
                room_type = None
                items = room

            if room_type is not None and room_type not in self.room_co_counts:
                self.room_co_counts[room_type] = {}
                room_tally[room_type] = {}

            # count individual popularity (for backup list) and setup dictionary
            for item in items:
                item_tally[item] = item_tally.get(item, 0) + 1
                if item not in self.co_counts:
                    self.co_counts[item] = {}

                if room_type is not None:
                    room_tally[room_type][item] = room_tally[room_type].get(item, 0) + 1
                    if item not in self.room_co_counts[room_type]:
                        self.room_co_counts[room_type][item] = {}

            # counting the pairs by looping through every combination of two items in the room
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item_a = items[i]
                    item_b = items[j]
                    
                    # add 1 to their co-occurrence tally in both directions
                    self.co_counts[item_a][item_b] = self.co_counts[item_a].get(item_b, 0) + 1
                    self.co_counts[item_b][item_a] = self.co_counts[item_b].get(item_a, 0) + 1

                    if room_type is not None:
                        r_counts = self.room_co_counts[room_type]
                        r_counts[item_a][item_b] = r_counts[item_a].get(item_b, 0) + 1
                        r_counts[item_b][item_a] = r_counts[item_b].get(item_a, 0) + 1
        
        # sort backup items by overall popularity
        sorted_items = sorted(item_tally.items(), key=lambda x: x[1], reverse=True)
        self.popular_backup = [item[0] for item in sorted_items]

        for r_type, tally in room_tally.items():
            sorted_r_items = sorted(tally.items(), key=lambda x: x[1], reverse=True)
            self.room_popular_backup[r_type] = [item[0] for item in sorted_r_items]

    def predict(self, partial_room, k=10, room_type=None):
        candidate_scores = {}
        items_in_room = partial_room

        if room_type is None and partial_room and isinstance(partial_room[0], str):
            candidate_rt = partial_room[0].replace("_", " ")
            if candidate_rt in self.room_co_counts:
                room_type = candidate_rt
                items_in_room = partial_room[1:]
        elif isinstance(room_type, str):
            room_type = room_type.replace("_", " ")

        co_counts = self.room_co_counts.get(room_type, self.co_counts)
        
        # tally up scores based on historical pairings
        for item in items_in_room:
            if item in co_counts:
                # look at every item this has ever been paired with
                for neighbor, count in co_counts[item].items():
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0) + count
                    
        # sort by the highest total co-occurrence score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for candidate, score in sorted_candidates:
            if candidate not in items_in_room:
                predictions.append(candidate)
            if len(predictions) == k:
                break
                
        # if we couldnt find enough pairs fill the rest with popular items for that room_type (then global)
        backup_list = self.room_popular_backup.get(room_type, self.popular_backup)
        if len(predictions) < k:
            for backup_item in backup_list:
                if backup_item not in items_in_room and backup_item not in predictions:
                    predictions.append(backup_item)
                if len(predictions) == k:
                    break

        if len(predictions) < k:
            for backup_item in self.popular_backup:
                if backup_item not in items_in_room and backup_item not in predictions:
                    predictions.append(backup_item)
                if len(predictions) == k:
                    break
                    
        return predictions
