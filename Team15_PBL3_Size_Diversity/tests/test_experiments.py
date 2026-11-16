from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ENSEMBLE_SIZES, FEATURE_SUBSETS, INTERACTION_SIZES
from src.data_preparation import conflicting_duplicate_groups, load_dataset, make_holdout
from src.diversity_metrics import diversity_summary
from src.ensemble_models import build_model


class TestPivotExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.X, cls.y = load_dataset(ROOT / "data" / "debrecen_dr.csv")
        cls.split = make_holdout(cls.X, cls.y)

    def test_verified_schema_and_duplicates(self):
        self.assertEqual(self.frame.shape, (1151, 19))
        self.assertEqual(self.frame.duplicated().sum(), 5)
        self.assertEqual(conflicting_duplicate_groups(self.X, self.y), 0)

    def test_holdout_has_no_duplicate_vector_leakage(self):
        self.assertTrue(set(self.split.groups_train).isdisjoint(self.split.groups_test))
        self.assertEqual((len(self.split.X_train), len(self.split.X_test)), (920, 231))

    def test_required_experimental_grids(self):
        self.assertEqual(ENSEMBLE_SIZES, [1, 5, 10, 25, 50, 100, 200, 300])
        self.assertEqual(INTERACTION_SIZES, [10, 25, 50, 100, 200, 300])
        self.assertEqual(FEATURE_SUBSETS, [1, 3, 6, 9, 18])

    def test_diversity_metric_known_example(self):
        predictions = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
        result = diversity_summary(predictions, np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(result["disagreement"], 0.5)
        self.assertAlmostEqual(result["double_fault"], 1 / 12)
        self.assertAlmostEqual(result["learner_accuracy"], 2 / 3)

    def test_algorithm_definitions(self):
        bag = build_model("Bagging", 10, {"max_depth": 12, "min_samples_leaf": 3, "max_samples": .8})
        rf = build_model("Random Forest", 10, {"max_depth": 12, "min_samples_leaf": 3, "max_samples": .8, "max_features": 3})
        gb = build_model("Gradient Boosting", 10, {"learning_rate": .1, "max_depth": 2, "subsample": .8})
        self.assertEqual(bag.max_features, 1.0)
        self.assertEqual(rf.max_features, 3)
        self.assertEqual(gb.n_estimators, 10)


if __name__ == "__main__":
    unittest.main()

