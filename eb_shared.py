"""
eb_shared.py
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
Shared utilities for all EB paper experiments.

All dataset scripts import from here to guarantee:
  - Identical EB circuit logic across all experiments
  - Identical representative selection
  - Identical QSVC runner
  - Identical metric computation
  - Identical summary/saving conventions
  - Consistent output naming

Nothing in this file is dataset-specific.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

# Classical baselines â€” identical across all datasets
CLASSICAL_MODELS = [
    ("KNN (k=3)",          KNeighborsClassifier(n_neighbors=3)),
    ("SVM (RBF)",          SVC(kernel="rbf", probability=True, random_state=0)),
    ("Random Forest",      RandomForestClassifier(n_estimators=120, random_state=0)),
    ("Logistic Regression",LogisticRegression(max_iter=2000, random_state=0)),
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1.  EB Circuits  (U_pair)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _append_zz_block(qc: QuantumCircuit, x: np.ndarray,
                     n: int, start: int, reps: int = 2) -> None:
    """ZZFeatureMap-style encoding block â€” used for the ad-hoc dataset."""
    for _ in range(reps):
        for i in range(n):
            qc.h(start + i)
            qc.rz(2.0 * x[i], start + i)
        for i in range(n - 1):
            for j in range(i + 1, n):
                angle = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                qc.cx(start + i, start + j)
                qc.rz(angle, start + j)
                qc.cx(start + i, start + j)


def u_pair_adhoc(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """
    EB circuit for the ad-hoc dataset.
    ZZFeatureMap encoding on each register + H + CNOT inter-register.
    """
    n  = len(x1)
    qc = QuantumCircuit(2 * n)
    _append_zz_block(qc, x1, n, start=0, reps=2)
    _append_zz_block(qc, x2, n, start=n, reps=2)
    qc.barrier()
    for i in range(n):
        qc.h(i)
        qc.cx(i, i + n)
    return qc


def u_pair_tabular(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """
    EB circuit for tabular datasets (breast cancer, fraud, Iris, Wine).
    RY angle encoding on each register + CNOT inter-register.
    """
    n  = len(x1)
    qc = QuantumCircuit(2 * n)
    for i, v in enumerate(x1):
        qc.ry(float(v), i)
    for i, v in enumerate(x2):
        qc.ry(float(v), n + i)
    qc.barrier()
    for i in range(n):
        qc.cx(i, n + i)
    return qc


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2.  Representative selection
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def select_representatives(
    X_class: np.ndarray,
    n_select: int,
    strategy: str = "kmeans",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Select n_select representatives from X_class.

    strategy : 'kmeans'  â†’ medoid of each cluster (default)
               'random'  â†’ uniform random sample
    """
    if len(X_class) <= n_select:
        return X_class.copy()

    if strategy == "random":
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_class), n_select, replace=False)
        return X_class[idx]

    if strategy == "kmeans":
        km     = KMeans(n_clusters=n_select, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_class)
        reps   = []
        for c in range(n_select):
            pts = X_class[labels == c]
            if len(pts) == 0:
                continue
            reps.append(pts[np.argmin(
                np.linalg.norm(pts - km.cluster_centers_[c], axis=1)
            )])
        return np.asarray(reps)

    raise ValueError(f"Unknown representative strategy: {strategy!r}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3.  EB state construction and scoring
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_class_statevectors(
    representatives: np.ndarray,
    circuit_fn: Callable,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build Î¨^(y): statevectors for all unordered representative pairs."""
    if len(representatives) < 2:
        raise ValueError("At least 2 representatives required.")
    svecs, pairs = [], []
    for i, j in combinations(range(len(representatives)), 2):
        qc = circuit_fn(representatives[i], representatives[j])
        svecs.append(Statevector.from_instruction(qc).data)
        pairs.append((i, j))
    return np.asarray(svecs), pairs


def build_test_statevectors(
    x: np.ndarray,
    representatives: np.ndarray,
    circuit_fn: Callable,
) -> np.ndarray:
    """Build Î¨_x^(y): statevectors for x paired with each representative."""
    return np.asarray([
        Statevector.from_instruction(circuit_fn(x, r)).data
        for r in representatives
    ])


def _aggregate(values: np.ndarray, mode: str) -> float:
    if mode == "mean":   return float(np.mean(values))
    if mode == "median": return float(np.median(values))
    if mode == "max":    return float(np.max(values))
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def predict_eb(
    X_test: np.ndarray,
    reps_neg: np.ndarray,
    reps_pos: np.ndarray,
    states_neg: np.ndarray,
    states_pos: np.ndarray,
    circuit_fn: Callable,
    agg_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run EB inference over X_test.

    Returns
    -------
    y_pred, margins, scores_neg, scores_pos  â€” all shape (len(X_test),)
    """
    y_pred, margins, s_neg, s_pos = [], [], [], []
    for x in X_test:
        t_neg = build_test_statevectors(x, reps_neg, circuit_fn)
        t_pos = build_test_statevectors(x, reps_pos, circuit_fn)
        fn = _aggregate(
            np.abs(t_neg @ states_neg.conj().T).ravel() ** 2, agg_mode)
        fp = _aggregate(
            np.abs(t_pos @ states_pos.conj().T).ravel() ** 2, agg_mode)
        y_pred.append(1 if fp >= fn else -1)
        margins.append(fp - fn)
        s_neg.append(fn)
        s_pos.append(fp)
    return (np.asarray(y_pred), np.asarray(margins),
            np.asarray(s_neg),  np.asarray(s_pos))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4.  QSVC runner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_qsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train and evaluate QSVC with ZZFeatureMap."""
    fmap   = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    kernel = FidelityQuantumKernel(feature_map=fmap)
    qsvc   = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    try:
        scores = np.asarray(qsvc.decision_function(X_test))
    except Exception:
        scores = np.asarray(y_pred, dtype=float)
    return np.asarray(y_pred), scores


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5.  Metrics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    out = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            out["ROC-AUC"] = roc_auc_score(
                (np.asarray(y_true) == 1).astype(int), scores)
        except Exception:
            out["ROC-AUC"] = np.nan
    else:
        out["ROC-AUC"] = np.nan
    return out


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6.  Results saving and summary
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def save_results(
    results: list[dict],
    raw_predictions: list[dict],
    dataset_prefix: str,
    output_dir: str,
) -> None:
    """
    Save raw results, raw predictions, formatted summary, and numeric summary.
    Also prints the final summary table.

    Files saved
    -----------
    <prefix>_raw_results.csv
    <prefix>_raw_predictions.csv
    <prefix>_summary.csv          (mean Â± std, human-readable)
    <prefix>_summary_numeric.csv  (separate mean and std columns)
    """
    os.makedirs(output_dir, exist_ok=True)

    df     = pd.DataFrame(results)
    raw_df = pd.DataFrame(raw_predictions)

    mean = df.groupby("Algorithm")[METRIC_COLS].mean() \
             .sort_values("Accuracy", ascending=False)
    std  = df.groupby("Algorithm")[METRIC_COLS].std().reindex(mean.index)

    # Formatted summary (mean Â± std)
    fmt = pd.DataFrame(index=mean.index)
    for col in METRIC_COLS:
        fmt[col] = [
            f"{mean.loc[a, col]:.4f} Â± {std.loc[a, col]:.4f}"
            if not np.isnan(mean.loc[a, col]) else "nan"
            for a in mean.index
        ]

    print("\n" + "=" * 78)
    print(f"FINAL SUMMARY â€” {dataset_prefix.upper()}")
    print("=" * 78)
    print(fmt.to_markdown())

    df.to_csv(
        os.path.join(output_dir, f"{dataset_prefix}_raw_results.csv"),
        index=False)
    raw_df.to_csv(
        os.path.join(output_dir, f"{dataset_prefix}_raw_predictions.csv"),
        index=False)
    fmt.reset_index().rename(columns={"index": "Algorithm"}).to_csv(
        os.path.join(output_dir, f"{dataset_prefix}_summary.csv"),
        index=False)

    num = mean.copy()
    for col in METRIC_COLS:
        num[f"{col} Std"] = std[col]
    num.reset_index().rename(columns={"index": "Algorithm"}).to_csv(
        os.path.join(output_dir, f"{dataset_prefix}_summary_numeric.csv"),
        index=False)

    print(f"\nSaved to {output_dir}/  [{dataset_prefix}_*]")


def save_bar_plots(
    results: list[dict],
    dataset_prefix: str,
    output_dir: str,
    dataset_title: str = "",
    figure_prefix: str = "EB",
) -> None:
    """Save Accuracy and F1-score bar charts."""
    df   = pd.DataFrame(results)
    mean = df.groupby("Algorithm")[METRIC_COLS].mean() \
             .sort_values("Accuracy", ascending=False)
    std  = df.groupby("Algorithm")[METRIC_COLS].std().reindex(mean.index)

    for metric in ["Accuracy", "F1-score"]:
        fig, ax = plt.subplots(figsize=(11, 5))
        x      = np.arange(len(mean))
        values = mean[metric].values
        errors = std[metric].fillna(0.0).values
        ax.bar(x, values, yerr=errors, capsize=4, color="#1f77b4", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(mean.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        title_base = dataset_title or dataset_prefix
        ax.set_title(f"{figure_prefix} - {title_base}: {metric}", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        fname = f"{dataset_prefix}_{metric.lower().replace('-','_')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close(fig)


def save_circuit_diagram(
    circuit_fn: Callable,
    example_x1: np.ndarray,
    example_x2: np.ndarray,
    dataset_prefix: str,
    output_dir: str,
) -> None:
    """Save text and PNG circuit diagram for the reference circuit."""
    qc = circuit_fn(example_x1, example_x2)
    txt_path = os.path.join(output_dir, f"{dataset_prefix}_circuit.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(qc.draw(output="text", fold=-1)))
    try:
        fig = qc.draw(output="mpl", fold=-1)
        fig.savefig(
            os.path.join(output_dir, f"{dataset_prefix}_circuit.png"),
            bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not save mpl circuit diagram: {exc}")

