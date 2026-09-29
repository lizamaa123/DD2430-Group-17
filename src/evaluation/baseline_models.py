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
    