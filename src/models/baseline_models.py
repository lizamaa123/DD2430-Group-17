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
                room_type = room["room_type"]
                items = room["item_ids"]
            else:
                room_type = None
                items = room

            for item in items:
                # if we have seen item before we add 1 to its tally
                if item in item_tally:
                    item_tally[item] = item_tally[item] + 1
                # o.w. it starts at 1
                else:
                    item_tally[item] = 1

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

        ranking = self.room_popularity.get(room_type, self.popularity_ranking)

        # loop through first item in our "popular items list"
        for item in ranking:
            # only recommend popular item if its not already in the room
            if item not in partial_room:
                predicitions.append(item)

            # stop guessing after k guesses 
            if len(predicitions) == k:
                break

        # fallback to global popularity if room_type ranking had fewer than k items
        if len(predicitions) < k:
            for item in self.popularity_ranking:
                if item not in partial_room and item not in predicitions:
                    predicitions.append(item)
                if len(predicitions) == k:
                    break

        return predicitions
    
class ItemContentBaseline:
    def __init__(self, item_id, item_descriptions):
        # model needs access to data through item ids
        self.item_id = np.array(item_id)
        
        # loading pre-trained SBERT
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # create embeddings of item descriptions
        self.item_embedding = self.model.encode(item_descriptions)

        # cache embeddings of room_type labels so we only encode each room_type once
        self.room_type_embeddings = {}

    def fit(self, training_rooms):
        # this baseline model doesn't need to learn from room groupings
        # so this method exist so the evaluation engine doesnt crash
        pass

    def predict(self, partial_room, k=10, room_type=None):
        # find embeddings for the items currently in the room
        present_item_rows = []
        for item in partial_room:
            if item in self.item_id:
                row_number = list(self.item_id).index(item)
                present_item_rows.append(row_number)
        
        if not present_item_rows:
            # if we do not have valid items we just return the first k catalog items
            return self.item_id[:k].tolist()
            
        present_item_embedding = self.item_embedding[present_item_rows]

        # if room_type is given, include its embedding in the room profile
        if room_type is not None:
            if room_type not in self.room_type_embeddings:
                clean_label = room_type.replace("_", " ")
                self.room_type_embeddings[room_type] = self.model.encode([clean_label])
            room_label_embedding = self.room_type_embeddings[room_type]
            present_item_embedding = np.vstack([present_item_embedding, room_label_embedding])

        # take the average of room embeddings to create a single "room profile" embedding
        room_embedding = np.mean(present_item_embedding, axis=0).reshape(1, -1)
        
        # caclulate similarity between the room profile and every item
        # convert from pytroch to numpy array
        similarities = self.model.similarity(room_embedding, self.item_embedding)[0].numpy()
        
        # sort by highest similarity score
        sorted_indices = np.argsort(similarities)[::-1]
        
        # pick the top k items that are not already in the room = predicitons 
        predictions = []
        for ids in sorted_indices:
            item = str(self.item_id[ids])
            if item not in partial_room:
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
                room_type = room["room_type"]
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
        co_counts = self.room_co_counts.get(room_type, self.co_counts)
        
        # tally up scores based on historical pairings
        for item in partial_room:
            if item in co_counts:
                # look at every item this has ever been paired with
                for neighbor, count in co_counts[item].items():
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0) + count
                    
        # sort by the highest total co-occurrence score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for candidate, score in sorted_candidates:
            if candidate not in partial_room:
                predictions.append(candidate)
            if len(predictions) == k:
                break
                
        # if we couldnt find enough pairs fill the rest with popular items for that room_type (then global)
        backup_list = self.room_popular_backup.get(room_type, self.popular_backup)
        if len(predictions) < k:
            for backup_item in backup_list:
                if backup_item not in partial_room and backup_item not in predictions:
                    predictions.append(backup_item)
                if len(predictions) == k:
                    break

        if len(predictions) < k:
            for backup_item in self.popular_backup:
                if backup_item not in partial_room and backup_item not in predictions:
                    predictions.append(backup_item)
                if len(predictions) == k:
                    break
                    
        return predictions