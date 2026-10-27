from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit

from . import RANDOM_STATE
from .data_validation import STATE_ORDER


def grouped_patient_split_indices(longitudinal: pd.DataFrame):
    """Return a deterministic split that keeps both eyes of a patient together."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=RANDOM_STATE)
    return next(splitter.split(longitudinal, groups=longitudinal.PtID))


def add_transitions(longitudinal: pd.DataFrame) -> pd.DataFrame:
    data = longitudinal.sort_values(["PtID", "RCEye", "VisitDt"]).copy()
    group = data.groupby(["PtID", "RCEye"], sort=False)
    data["previous_state"] = group.severity_state.shift(1)
    data["next_state"] = group.severity_state.shift(-1)
    data["next_visit"] = group.Visit.shift(-1)
    data["next_date"] = group.VisitDt.shift(-1)
    data["transition_days"] = (data.next_date - data.VisitDt).dt.days
    return data


def transition_matrices(longitudinal: pd.DataFrame):
    data = add_transitions(longitudinal)
    transitions = data.loc[data.next_state.notna()].copy()
    counts = pd.crosstab(transitions.severity_state, transitions.next_state).reindex(
        index=STATE_ORDER, columns=STATE_ORDER, fill_value=0
    )
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    transitions["from_order"] = transitions.severity_state.map({s: i for i, s in enumerate(STATE_ORDER)})
    transitions["to_order"] = transitions.next_state.map({s: i for i, s in enumerate(STATE_ORDER)})
    transitions["change"] = np.select(
        [transitions.to_order.gt(transitions.from_order), transitions.to_order.lt(transitions.from_order)],
        ["Progression", "Regression"], default="Stable",
    )
    return transitions, counts, probs


def bootstrap_transition_intervals(
    longitudinal: pd.DataFrame, repeats: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    patient_ids = longitudinal.PtID.unique()
    samples = []
    for _ in range(repeats):
        chosen = rng.choice(patient_ids, len(patient_ids), replace=True)
        pieces = [longitudinal.loc[longitudinal.PtID.eq(pid)].assign(_boot=i) for i, pid in enumerate(chosen)]
        boot = pd.concat(pieces, ignore_index=True)
        boot["PtID_boot"] = boot.PtID.astype(str) + "_" + boot._boot.astype(str)
        temp = boot.rename(columns={"PtID": "PtID_original", "PtID_boot": "PtID"})
        _, _, matrix = transition_matrices(temp)
        samples.append(matrix.to_numpy())
    arr = np.stack(samples)
    low = pd.DataFrame(np.nanquantile(arr, .025, axis=0), index=STATE_ORDER, columns=STATE_ORDER)
    high = pd.DataFrame(np.nanquantile(arr, .975, axis=0), index=STATE_ORDER, columns=STATE_ORDER)
    return low, high


def state_distribution_evolution(longitudinal: pd.DataFrame, probs: pd.DataFrame, steps: int = 4) -> pd.DataFrame:
    baseline = longitudinal.loc[longitudinal.Visit.eq("Baseline"), "severity_state"].value_counts(normalize=True).reindex(STATE_ORDER, fill_value=0).to_numpy()
    matrix = probs.fillna(0).to_numpy()
    rows = []
    current = baseline
    for step in range(steps + 1):
        rows.extend({"step": step, "state": state, "probability": current[i]} for i, state in enumerate(STATE_ORDER))
        current = current @ matrix
    return pd.DataFrame(rows)


def transition_sensitivity(probs: pd.DataFrame) -> pd.DataFrame:
    base = probs.fillna(0).to_numpy()
    scenarios = {"Empirical": base.copy(), "Higher progression (hypothetical)": base.copy()}
    high = scenarios["Higher progression (hypothetical)"]
    for i in range(len(STATE_ORDER) - 1):
        transferable = min(.10, high[i, i])
        high[i, i] -= transferable
        high[i, i + 1] += transferable
    rows = []
    start = np.array([0, 1, 0, 0], dtype=float)
    for scenario, matrix in scenarios.items():
        for step in [1, 2, 4]:
            dist = start @ np.linalg.matrix_power(matrix, step)
            rows.extend({"scenario": scenario, "step": step, "state": s, "probability": dist[j]} for j, s in enumerate(STATE_ORDER))
    return pd.DataFrame(rows)


def compare_markov_orders(longitudinal: pd.DataFrame) -> pd.DataFrame:
    """Patient-grouped holdout comparison; second order backs off on unseen pairs."""
    train_idx, test_idx = grouped_patient_split_indices(longitudinal)
    train = longitudinal.iloc[train_idx].copy()
    test = longitudinal.iloc[test_idx].copy()
    train_t = add_transitions(train).dropna(subset=["next_state"])
    first_counts = pd.crosstab(train_t.severity_state, train_t.next_state).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    first_probs = (first_counts + 1).div((first_counts + 1).sum(axis=1), axis=0)
    train_trip = add_transitions(train).dropna(subset=["previous_state", "next_state"])
    second_counts = train_trip.groupby(["previous_state", "severity_state", "next_state"]).size()
    test_eval = add_transitions(test).dropna(subset=["previous_state", "next_state"]).copy()
    true = test_eval.next_state.tolist()
    true_index = [STATE_ORDER.index(state) for state in true]
    p1, p2, seen = [], [], []
    for row in test_eval.itertuples():
        first = first_probs.loc[row.severity_state].to_numpy()
        p1.append(first)
        context = (row.previous_state, row.severity_state)
        if context in second_counts.index.droplevel(2):
            values = second_counts.loc[context].reindex(STATE_ORDER, fill_value=0).to_numpy() + 1
            p2.append(values / values.sum()); seen.append(True)
        else:
            p2.append(first); seen.append(False)
    results = []
    for order, pred in [("First-order", np.array(p1)), ("Second-order with first-order backoff", np.array(p2))]:
        labels = np.array(STATE_ORDER)[pred.argmax(axis=1)]
        results.append({"model": order, "test_transitions": len(true),
                        "accuracy": accuracy_score(true, labels),
                        "log_loss": log_loss(true_index, pred, labels=list(range(len(STATE_ORDER)))),
                        "second_order_context_coverage": np.mean(seen) if order.startswith("Second") else np.nan,
                        "train_patients": train.PtID.nunique(), "test_patients": test.PtID.nunique()})
    return pd.DataFrame(results)

