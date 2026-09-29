import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PopularityPrior():
    def __init__(self):
        # score final sorted list of popular items
        self.popularity_ranking = []

    def fit(self, training_rooms):
        item_tally = {}

        # loop through every room and then loop through all items
        for room in training_rooms:
            for item in room:
                # if we have seen item before we add 1 to its tally
                if item in item_tally:
                    item_tally[item] = item_tally[item] + 1
                # o.w. it starts at 1
                else:
                    item_tally[item] = 1


        sorted_items = sorted(item_tally.items(), key=lambda x: x[1], reverse=True)

        self.popularity_ranking = [item[0] for item in sorted_items]

    def predict(self, partial_room, k=10):
        predicitions = []

        # loop through first item in our "popular items list"
        for item in self.popularity_ranking:
            # only recommend popular item if its not already in the room
            if item not in partial_room:
                predicitions.append(item)

            # stop guessing after k guesses 
            if len(predicitions) == k:
                break

        return predicitions
    
class ItemContentBaseline:
    def __init__(self, item_id, item_descriptions):
        # model needs access to data through item ids
        self.item_id = np.array(item_id)
        
        # loading pre-trained SBERT
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # create embeddings of item descriptions
        self.item_embedding = self.model.encode(item_descriptions)

    def fit(self, training_rooms):
        # this baseline model doesn't need to learn from room groupings
        # so this method exist so the evaluation engine doesnt crash
        pass

    def predict(self, partial_room, k=10):
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

        # take the average of room embeddings to create a single "room profile" embedding
        room_embedding = np.mean(present_item_embedding, axis=0).reshape(1, -1)
        
        # caclulate similarity between the room profile and every item
        similarities = cosine_similarity(room_embedding, self.item_embedding)[0]
        
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