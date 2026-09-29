import unittest
from src.evaluation.baseline_models import PopularityPrior

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

if __name__ == '__main__':
    unittest.main()