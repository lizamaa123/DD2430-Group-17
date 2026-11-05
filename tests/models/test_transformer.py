import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.transformer import FurnitureEmbeddingPredictor, calculate_running_average


class TestTransformerLivePlot(unittest.TestCase):
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

    def test_fit_with_live_plot_disabled(self):
        # Create a lightweight model on CPU for testing
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, device="cpu")

        # Create dummy batch
        batch_size = 4
        room_len = 5
        dummy_batch = {
            "x": torch.randn(batch_size, room_len, 16),
            "valid_mask": torch.ones(batch_size, room_len, dtype=torch.bool),
        }
        train_loader = [dummy_batch, dummy_batch]

        losses = model.fit(train_loader, epochs=1, live_plot=False)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(isinstance(l, float) for l in losses))

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.pause")
    def test_fit_with_live_plot_enabled(self, mock_pause, mock_show):
        model = FurnitureEmbeddingPredictor(embed_dim=16, hid_dim=32, nhead=2, num_layers=1, device="cpu")

        batch_size = 2
        room_len = 3
        dummy_batch = {
            "x": torch.randn(batch_size, room_len, 16),
            "valid_mask": torch.ones(batch_size, room_len, dtype=torch.bool),
        }
        train_loader = [dummy_batch, dummy_batch, dummy_batch]

        losses = model.fit(train_loader, epochs=1, live_plot=True, plot_every=1)
        self.assertEqual(len(losses), 3)
        self.assertTrue(mock_pause.called)
        self.assertTrue(mock_show.called)


if __name__ == "__main__":
    unittest.main()
