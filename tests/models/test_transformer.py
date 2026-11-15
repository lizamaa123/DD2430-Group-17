import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.transformer import (
    FurnitureEmbeddingPredictor,
    TransformerEvaluator,
    calculate_running_average,
)


class TestTransformer(unittest.TestCase):
    def test_calculate_running_average(self):
        # Empty
        self.assertEqual(len(calculate_running_average([])), 0)

        # Single value
        avg1 = calculate_running_average([0.5])
        self.assertEqual(len(avg1), 1)
        self.assertAlmostEqual(avg1[0], 0.5)

        # Multiple values below window
        losses = [1.0, 0.5, 0.3]
        avg = calculate_running_average(losses, window_size=5)
        self.assertEqual(len(avg), 3)
        self.assertAlmostEqual(avg[0], 1.0)
        self.assertAlmostEqual(avg[1], 0.75)
        self.assertAlmostEqual(avg[2], 0.6)

        # Values above window
        losses_long = [1.0] * 10 + [2.0] * 10
        avg_long = calculate_running_average(losses_long, window_size=5)
        self.assertEqual(len(avg_long), 20)
        self.assertAlmostEqual(avg_long[-1], 2.0)

    def test_forward_single_and_multi_head(self):
        # Single head
        model_1 = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=1, device="cpu")
        x = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out_1 = model_1.forward(x, mask, mask_index=1)
        self.assertEqual(out_1.shape, (2, 16))

        # Multi head
        model_3 = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=3, device="cpu")
        out_3 = model_3.forward(x, mask, mask_index=1)
        self.assertEqual(out_3.shape, (2, 3, 16))

        preds, conf = model_3.forward(x, mask, mask_index=1, return_confidence=True)
        self.assertEqual(preds.shape, (2, 3, 16))
        self.assertEqual(conf.shape, (2, 3))

    def test_fit_multi_head_wta_infonce(self):
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=3, device="cpu")
        batch_size = 4
        room_len = 5
        dummy_batch = {
            "x": torch.randn(batch_size, room_len, 16),
            "valid_mask": torch.ones(batch_size, room_len, dtype=torch.bool),
        }
        train_loader = [dummy_batch, dummy_batch]

        losses = model.fit(train_loader, epochs=1, live_plot=False, loss_type="infonce", temperature=0.1, conf_weight=0.2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(isinstance(l, float) for l in losses))

    def test_fit_multi_head_wta_cosine(self):
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=3, device="cpu")
        batch_size = 4
        room_len = 5
        dummy_batch = {
            "x": torch.randn(batch_size, room_len, 16),
            "valid_mask": torch.ones(batch_size, room_len, dtype=torch.bool),
        }
        train_loader = [dummy_batch]

        losses = model.fit(train_loader, epochs=1, live_plot=False, loss_type="cosine", conf_weight=0.2)
        self.assertEqual(len(losses), 1)
        self.assertTrue(all(isinstance(l, float) for l in losses))

    def test_fit_invalid_loss_type(self):
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, device="cpu")
        dummy_batch = {
            "x": torch.randn(2, 3, 16),
            "valid_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        with self.assertRaises(ValueError):
            model.fit([dummy_batch], epochs=1, live_plot=False, loss_type="unsupported")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.pause")
    def test_fit_with_live_plot_enabled(self, mock_pause, mock_show):
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=3, device="cpu")

        batch_size = 2
        room_len = 3
        dummy_batch = {
            "x": torch.randn(batch_size, room_len, 16),
            "valid_mask": torch.ones(batch_size, room_len, dtype=torch.bool),
        }
        train_loader = [dummy_batch, dummy_batch, dummy_batch]

        losses = model.fit(train_loader, epochs=1, live_plot=True, plot_every=1, loss_type="infonce")
        self.assertEqual(len(losses), 3)
        self.assertTrue(mock_pause.called)
        self.assertTrue(mock_show.called)

    @patch("src.models.transformer.SentenceTransformer")
    def test_evaluator_multi_head(self, mock_sbert_cls):
        mock_sbert = MagicMock()
        mock_sbert.encode.return_value = torch.randn(2, 16)
        mock_sbert_cls.return_value = mock_sbert

        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, num_candidates=3, device="cpu")
        item_descs = ["chair", "table", "bed", "lamp"]
        item_ids = [101, 102, 103, 104]
        item_embeddings = torch.randn(4, 16)

        evaluator = TransformerEvaluator(
            model=model,
            item_descriptions=item_descs,
            item_ids=item_ids,
            item_embeddings=item_embeddings,
            return_strings=True,
            aggregation="max",
        )

        preds = evaluator.predict(["chair", "table"], k=2)
        self.assertEqual(len(preds), 2)
        self.assertNotIn("chair", preds)
        self.assertNotIn("table", preds)

        choices = evaluator.predict_choices(["chair", "table"], k_per_head=1)
        self.assertEqual(len(choices), 3)
        self.assertTrue("head" in choices[0])
        self.assertTrue("confidence" in choices[0])
        self.assertTrue("predictions" in choices[0])


if __name__ == "__main__":
    unittest.main()
