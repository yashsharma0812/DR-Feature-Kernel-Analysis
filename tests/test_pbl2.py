from pathlib import Path
import unittest

import numpy as np

from src.pbl2.data_loader import discover_archive
from src.pbl2.data_validation import STATE_ORDER, build_longitudinal_dataset, severity_state
from src.pbl2.markov import grouped_patient_split_indices, transition_matrices
from src.pbl2.preprocessing import CLUSTER_FEATURES, build_baseline_dataset


ROOT = Path(__file__).resolve().parents[1]


class TestProtocolAAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = discover_archive(ROOT)
        cls.baseline, _ = build_baseline_dataset(cls.archive)
        cls.longitudinal, cls.audit = build_longitudinal_dataset(cls.archive)

    def test_state_mapping_and_ungradable(self):
        self.assertEqual(severity_state(3), STATE_ORDER[0])
        self.assertEqual(severity_state(5), STATE_ORDER[1])
        self.assertEqual(severity_state(7), STATE_ORDER[2])
        self.assertEqual(severity_state(12), STATE_ORDER[3])
        self.assertIsNone(severity_state(90))

    def test_unique_analysis_keys(self):
        self.assertFalse(self.baseline.duplicated(["PtID", "Eye"]).any())
        self.assertFalse(self.longitudinal.duplicated(["PtID", "RCEye", "Visit"]).any())

    def test_no_severity_or_future_information_in_cluster_features(self):
        forbidden = {"dro", "severity_state", "original_dr_grade", "VisitDt", "Visit"}
        self.assertTrue(forbidden.isdisjoint(CLUSTER_FEATURES))

    def test_transition_totals_and_rows(self):
        transitions, counts, probs = transition_matrices(self.longitudinal)
        expected = sum(max(n - 1, 0) for n in self.longitudinal.groupby(["PtID", "RCEye"]).size())
        self.assertEqual(len(transitions), expected)
        self.assertEqual(int(counts.to_numpy().sum()), expected)
        self.assertTrue(np.allclose(probs.sum(axis=1).to_numpy(), 1.0))

    def test_chronological_order(self):
        for _, sequence in self.longitudinal.groupby(["PtID", "RCEye"]):
            self.assertTrue(sequence.VisitDt.is_monotonic_increasing)

    def test_patient_group_split_has_no_leakage(self):
        train_idx, test_idx = grouped_patient_split_indices(self.longitudinal)
        train_patients = set(self.longitudinal.iloc[train_idx].PtID)
        test_patients = set(self.longitudinal.iloc[test_idx].PtID)
        self.assertTrue(train_patients.isdisjoint(test_patients))


if __name__ == "__main__":
    unittest.main()

