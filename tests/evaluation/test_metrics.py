import math
import unittest
from src.evaluation.metrics import recall_at_k, ndcg_at_k


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.predictions = ["item_a", "item_b", "item_c", "item_d", "item_e"]

    def test_recall_hit(self):
        """Recall should return 1.0 when target is within top-k."""
        score = recall_at_k(self.predictions, target_item="item_b", k=3)
        self.assertEqual(score, 1.0)

    def test_recall_miss(self):
        """Recall should return 0.0 when target is not present."""
        score = recall_at_k(self.predictions, target_item="item_z", k=5)
        self.assertEqual(score, 0.0)

    def test_recall_outside_k_cutoff(self):
        """Recall should return 0.0 if target appears after k cut-off."""
        # item_c is at rank 3 but k=2
        score = recall_at_k(self.predictions, target_item="item_c", k=2)
        self.assertEqual(score, 0.0)

    def test_ndcg_rank_1(self):
        """NDCG should return a perfect 1.0 for a rank-1 prediction."""
        score = ndcg_at_k(self.predictions, target_item="item_a", k=5)
        self.assertEqual(score, 1.0)

    def test_ndcg_discounted_rank(self):
        """NDCG should discount according to the position."""
        # item_b is at rank 2: 1 / log2(2 + 1)
        expected_score = 1.0 / math.log2(2 + 1)
        score = ndcg_at_k(self.predictions, target_item="item_b", k=5)
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_ndcg_miss(self):
        """NDCG should return 0.0 when target is absent."""
        score = ndcg_at_k(self.predictions, target_item="item_z", k=5)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()