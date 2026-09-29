import unittest
from unittest.mock import patch
from src.evaluation.evaluator import LeaveOneOutEvaluator

class DummyModel:
    def predict(self, partial_room, k):
        return ["fake_guess_1", "fake_guess_2"]

class TestLeaveOneOutEvaluator(unittest.TestCase):
    # We patch the metric functions where they are imported inside evaluator.py
    @patch("src.evaluation.evaluator.recall_at_k")
    @patch("src.evaluation.evaluator.ndcg_at_k")
    
    def test_evaluate_calculates_averages_and_skips_short_rooms(self, mock_ndcg, mock_recall):
        # force the mocked metrics to always return specific scores
        mock_recall.return_value = 1.0
        mock_ndcg.return_value = 0.5

        # setup mock test data: two valid rooms, one invalid room
        test_rooms = [
            ["bed", "nightstand", "lamp"],  # Valid: length 3
            ["sofa", "table"],              # Valid: length 2
            ["chair"]                       # Invalid: length 1 (must be skipped)
        ]
        
        # initialize evaluator and dummy model
        evaluator = LeaveOneOutEvaluator(k=2)
        model = DummyModel()
        avg_recall, avg_ndcg = evaluator.evaluate(model, test_rooms)

        # Two valid rooms were processed. 
        # Total recall = 1.0 + 1.0 = 2.0. Average = 2.0 / 2 = 1.0
        # Total ndcg = 0.5 + 0.5 = 1.0. Average = 1.0 / 2 = 0.5
        self.assertEqual(avg_recall, 1.0)
        self.assertEqual(avg_ndcg, 0.5)
        
        # The metric functions should have only been called 2 times, not 3
        self.assertEqual(mock_recall.call_count, 2)
        self.assertEqual(mock_ndcg.call_count, 2)

if __name__ == "__main__":
    unittest.main()