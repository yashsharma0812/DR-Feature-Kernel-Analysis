from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import initial_models, parameter_grids
from src.preprocessing import FEATURES, load_dataset, make_holdout_split


class TestPBL3Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.X, cls.y = load_dataset(ROOT / "data" / "debrecen_dr.csv")
        cls.split = make_holdout_split(cls.X, cls.y)

    def test_verified_dataset_shape(self):
        self.assertEqual(self.frame.shape, (1151, 19))
        self.assertEqual(len(FEATURES), 18)
        self.assertEqual(set(self.y.unique()), {0, 1})

    def test_identical_vectors_do_not_cross_holdout(self):
        self.assertTrue(set(self.split.groups_train).isdisjoint(self.split.groups_test))

    def test_split_is_approximately_80_20_and_stratified(self):
        self.assertLess(abs(len(self.split.X_test) / len(self.X) - 0.2), 0.02)
        self.assertLess(abs(self.split.y_train.mean() - self.split.y_test.mean()), 0.03)

    def test_finalized_models_only(self):
        models = initial_models()
        self.assertEqual(set(models), {"Bagging", "Gradient Boosting", "Random Forest"})
        self.assertEqual(models["Bagging"].max_features, 1.0)
        self.assertTrue(models["Bagging"].bootstrap)
        self.assertTrue(models["Random Forest"].bootstrap)
        self.assertEqual(models["Random Forest"].max_features, "sqrt")

    def test_comparable_search_budgets(self):
        sizes = [int(np.prod([len(values) for values in grid.values()])) for grid in parameter_grids().values()]
        self.assertEqual(sizes, [24, 24, 24])


if __name__ == "__main__":
    unittest.main()

