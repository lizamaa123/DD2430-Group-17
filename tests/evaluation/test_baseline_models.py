import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.evaluation.baseline_models import PopularityPrior, ItemContentBaseline

class TestPopularityPrior(unittest.TestCase):
    def setUp(self):
        """This runs before every single test to set up fresh fake data."""
        self.model = PopularityPrior()
        # "lamp" appears 2 times everything else only once
        self.mock_rooms = [
            ["bed", "lamp"], 
            ["chair", "table", "lamp"], 
            ["rug"]
        ]
        self.model.fit(self.mock_rooms)

    def test_highest_frequency_is_first(self):
        """Checks if the most popular item is at the top of the ranking."""
        # lamp should be the #1 most popular item
        self.assertEqual(self.model.popularity_ranking[0], "lamp")

    def test_target_item_is_excluded(self):
        """Checks if the model successfully avoids recommending an item already in the room."""
        # If the room already has a lamp the guesses should NOT include lamp
        guesses = self.model.predict(partial_room=["lamp"], k=2)
        self.assertNotIn("lamp", guesses)

    def test_correct_number_of_predictions(self):
        """Checks if the model respects the 'k' limit."""
        # We ask for exactly 3 guesses
        guesses = self.model.predict(partial_room=["rug"], k=3)
        self.assertEqual(len(guesses), 3)

class TestItemContentBaseline(unittest.TestCase):

    # @patch intercepts SentenceTransformer when the class tries to load it
    # replacing it with our MockTransformer so we don't download the real AI model
    @patch('src.evaluation.baseline_models.SentenceTransformer')
    def setUp(self, MockTransformer):
        self.item_id = ["wooden_bed", "wooden_nightstand", "metal_lamp", "metal_desk"]
        self.item_descriptions = ["A wooden bed", "A wooden nightstand", "A metal lamp", "A metal desk"]
        
        mock_model_instance = MagicMock()
        mock_model_instance.encode.return_value = np.array([
            [1.0, 0.0],  # wooden_bed 
            [0.9, 0.0],  # wooden_nightstand 
            [0.1, 1.0],  # metal_lamp 
            [0.0, 0.9]   # metal_desk
        ])
        MockTransformer.return_value = mock_model_instance
        
        self.baseline = ItemContentBaseline(self.item_id, self.item_descriptions)

    def test_predicts_most_similar_items(self):
        """It should recommend the 'wooden_nightstand' when given a 'wooden_bed'."""
        partial_room = ["wooden_bed"]
        
        predictions = self.baseline.predict(partial_room, k=1)
        
        self.assertEqual(predictions, ["wooden_nightstand"])

    def test_excludes_items_already_in_room(self):
        """It should never recommend an item that is already in the partial room."""
        partial_room = ["wooden_bed", "wooden_nightstand"]
        
        predictions = self.baseline.predict(partial_room, k=2)
        
        # even though the bed and nightstand are the most mathematically similar 
        # to the room embedding, they must be skipped.
        self.assertNotIn("wooden_bed", predictions)
        self.assertNotIn("wooden_nightstand", predictions)
        
        # it is then forced to recommend the metal items instead
        self.assertEqual(predictions, ["metal_lamp", "metal_desk"])

    def test_handles_unknown_items_safely(self):
        """If the room has no known items, it should return the first k items."""
        partial_room = ["unknown_alien_artifact"]
        
        predictions = self.baseline.predict(partial_room, k=2)
        
        # it defaults to returning the first 2 items in the catalog
        self.assertEqual(predictions, ["wooden_bed", "wooden_nightstand"])

if __name__ == '__main__':
    unittest.main()