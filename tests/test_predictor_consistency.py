import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.predict import Predictor


class PredictorConsistencyTest(unittest.TestCase):
    def test_stabilize_prediction_prefers_most_common_recent_label(self):
        predictor = Predictor(model_wrapper=None, class_names=["paper", "plastic"], img_size=224)

        recent_predictions = ["plastic", "paper", "paper", "paper"]
        stabilized = predictor.stabilize_prediction(recent_predictions, window_size=3)

        self.assertEqual(stabilized, "paper")

    def test_canonical_label_normalizes_display_and_raw_names(self):
        predictor = Predictor(model_wrapper=None, class_names=["glass", "plastic"], img_size=224)

        self.assertEqual(predictor.canonical_label("Plastic"), "plastic")
        self.assertEqual(predictor.canonical_label("plastic"), "plastic")
        self.assertEqual(predictor.canonical_label("plastic "), "plastic")


if __name__ == "__main__":
    unittest.main()
