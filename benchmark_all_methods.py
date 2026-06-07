"""
benchmark_all_methods.py
===========================================================================
Unified benchmark suite for:
  - EB (Mean / Median / Max)
  - CPC (Cosine / RBF x Mean / Median / Max)
  - QSVC on representatives (default) or full train
  - Existing classical baselines
  - Additional reviewer-requested baselines

Datasets covered:
  - adhoc
  - breastcancer
  - fraud
  - iris
  - wine
  - phase_ising
  - circuit_outputs
  - entanglement
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from qiskit import QuantumCircuit

import experiment_adhoc
import experiment_breastcancer
import experiment_fraud
import experiment_iris
import experiment_wine
import ising_eb_experiment as quantum_exp

from eb_shared import (
    CLASSICAL_MODELS,
    METRIC_COLS,
    build_class_statevectors,
    compute_metrics,
    predict_eb,
    run_qsvc,
    save_circuit_diagram,
    save_results,
    select_representatives,
    u_pair_adhoc,
    u_pair_tabular,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "benchmark_all_methods"

CLASSICAL_DATASETS = [
    "adhoc",
    "breastcancer",
    "fraud",
    "iris",
    "wine",
]

QUANTUM_DATASETS = [
    "phase_ising",
    "circuit_outputs",
    "entanglement",
]

ALL_DATASETS = CLASSICAL_DATASETS + QUANTUM_DATASETS


def aggregate_values(values: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(u, v) / denom)


def rbf_similarity(u: np.ndarray, v: np.ndarray, gamma: float) -> float:
    diff = u - v
    return float(np.exp(-gamma * np.dot(diff, diff)))


def estimator_scores(estimator, X: np.ndarray, positive_label: int = 1) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(getattr(estimator, "classes_", []))
        if positive_label in classes:
            return np.asarray(proba[:, classes.index(positive_label)], dtype=float)
        return np.asarray(proba[:, -1], dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim == 1:
            return scores
        return scores[:, -1]
    return np.asarray(estimator.predict(X), dtype=float)


def nearest_centroid_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_centroid = X_train[y_train == 1].mean(axis=0)
    neg_centroid = X_train[y_train == -1].mean(axis=0)
    pos_dist = np.linalg.norm(X_test - pos_centroid, axis=1)
    neg_dist = np.linalg.norm(X_test - neg_centroid, axis=1)
    scores = neg_dist - pos_dist
    y_pred = np.where(pos_dist <= neg_dist, 1, -1)
    return y_pred.astype(int), scores.astype(float)


def pair_feature_vector(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.concatenate([x, r, np.abs(x - r), x * r])


def train_pairwise_estimator(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_reps: dict[int, np.ndarray],
):
    pair_X, pair_y = [], []
    for x, y in zip(X_train, y_train):
        for label, reps in class_reps.items():
            target = 1 if y == label else 0
            for r in reps:
                pair_X.append(pair_feature_vector(x, r))
                pair_y.append(target)
    estimator.fit(np.asarray(pair_X), np.asarray(pair_y))
    return estimator


def predict_pairwise_estimator(
    estimator,
    X_test: np.ndarray,
    class_reps: dict[int, np.ndarray],
    aggregation: str = "max",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_pred, margins, score_neg, score_pos = [], [], [], []
    for x in X_test:
        class_scores = {}
        for label, reps in class_reps.items():
            pair_X = np.asarray([pair_feature_vector(x, r) for r in reps])
            pair_scores = estimator_scores(estimator, pair_X, positive_label=1)
            class_scores[label] = aggregate_values(pair_scores, aggregation)
        neg = class_scores[-1]
        pos = class_scores[1]
        y_pred.append(1 if pos >= neg else -1)
        margins.append(pos - neg)
        score_neg.append(neg)
        score_pos.append(pos)
    return (
        np.asarray(y_pred, dtype=int),
        np.asarray(margins, dtype=float),
        np.asarray(score_neg, dtype=float),
        np.asarray(score_pos, dtype=float),
    )


def cpc_predict(
    X_test: np.ndarray,
    class_reps: dict[int, np.ndarray],
    similarity: str,
    aggregation: str,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_pred, margins, score_neg, score_pos = [], [], [], []
    for x in X_test:
        class_scores = {}
        for label, reps in class_reps.items():
            pair_vectors = [
                np.concatenate([reps[i], reps[j]])
                for i, j in combinations(range(len(reps)), 2)
            ]
            if not pair_vectors:
                pair_vectors = [np.concatenate([reps[0], reps[0]])]
            sims = []
            for r in reps:
                test_pair = np.concatenate([x, r])
                for pair_vec in pair_vectors:
                    if similarity == "cosine":
                        sims.append(cosine_similarity(test_pair, pair_vec))
                    else:
                        sims.append(rbf_similarity(test_pair, pair_vec, gamma))
            class_scores[label] = aggregate_values(np.asarray(sims), aggregation)
        neg = class_scores[-1]
        pos = class_scores[1]
        y_pred.append(1 if pos >= neg else -1)
        margins.append(pos - neg)
        score_neg.append(neg)
        score_pos.append(pos)
    return (
        np.asarray(y_pred, dtype=int),
        np.asarray(margins, dtype=float),
        np.asarray(score_neg, dtype=float),
        np.asarray(score_pos, dtype=float),
    )


def append_result_rows(
    results: list[dict],
    raw_predictions: list[dict],
    dataset_key: str,
    partition: int,
    algorithm: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    train_scope: str,
    score_neg: np.ndarray | None = None,
    score_pos: np.ndarray | None = None,
) -> None:
    metrics = compute_metrics(y_true, y_pred, scores=scores)
    metrics.update(
        {
            "Algorithm": algorithm,
            "Partition": partition,
            "Dataset": dataset_key,
            "Train Scope": train_scope,
        }
    )
    results.append(metrics)

    neg_scores = score_neg if score_neg is not None else np.full(len(y_true), np.nan)
    pos_scores = score_pos if score_pos is not None else np.full(len(y_true), np.nan)
    for row_idx, (yt, yp, sc, sn, sp) in enumerate(
        zip(y_true, y_pred, scores, neg_scores, pos_scores, strict=False)
    ):
        raw_predictions.append(
            {
                "Dataset": dataset_key,
                "Partition": partition,
                "Row": row_idx,
                "Algorithm": algorithm,
                "Train Scope": train_scope,
                "y_true": int(yt),
                "y_pred": int(yp),
                "score_margin": float(sc),
                "score_neg": float(sn),
                "score_pos": float(sp),
            }
        )


def build_common_estimators(X_train: np.ndarray) -> list[tuple[str, object]]:
    n_samples = len(X_train)
    n_features = X_train.shape[1]
    n_neighbors = max(1, min(3, n_samples))
    n_components = max(2, min(64, n_samples))
    gamma = 1.0 / max(1, n_features)

    estimators = [(name, clone(model)) for name, model in CLASSICAL_MODELS]
    estimators.extend(
        [
            (
                "NCA + KNN",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("nca", NeighborhoodComponentsAnalysis(random_state=0)),
                        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
                    ]
                ),
            ),
            (
                "Nystroem + LogReg",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "nystroem",
                            Nystroem(
                                kernel="rbf",
                                gamma=gamma,
                                n_components=n_components,
                                random_state=0,
                            ),
                        ),
                        ("logreg", LogisticRegression(max_iter=2000, random_state=0)),
                    ]
                ),
            ),
        ]
    )
    return estimators


def summarize_numeric(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    mean = df.groupby("Algorithm")[METRIC_COLS].mean()
    std = df.groupby("Algorithm")[METRIC_COLS].std()
    out = mean.copy()
    for col in METRIC_COLS:
        out[f"{col} Std"] = std[col]
    return out.reset_index()


def save_global_outputs(
    all_results: dict[str, list[dict]],
    output_dir: Path,
    config: dict,
) -> None:
    if not all_results:
        return

    summary_frames = []
    for dataset_key, dataset_results in all_results.items():
        if not dataset_results:
            continue
        frame = summarize_numeric(dataset_results)
        frame.insert(0, "Dataset", dataset_key)
        summary_frames.append(frame)

    if not summary_frames:
        return

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.to_csv(output_dir / "benchmark_all_datasets_summary_numeric.csv", index=False)

    auc_pivot = summary_df.pivot(index="Algorithm", columns="Dataset", values="ROC-AUC")
    auc_pivot.to_csv(output_dir / "benchmark_all_datasets_auc_pivot.csv")

    with open(output_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_dataset_comparison_plots(
    dataset_key: str,
    dataset_title: str,
    results: list[dict],
    raw_predictions: list[dict],
    output_dir: Path,
) -> None:
    if not results:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    mean = df.groupby("Algorithm")[METRIC_COLS].mean().sort_values("ROC-AUC", ascending=False)
    std = df.groupby("Algorithm")[METRIC_COLS].std().reindex(mean.index)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(mean))
    width = 0.25
    metrics = ["Accuracy", "F1-score", "ROC-AUC"]
    colors = ["#1f77b4", "#2a9d8f", "#e76f51"]
    for idx, (metric, color) in enumerate(zip(metrics, colors, strict=False)):
        values = mean[metric].values
        errors = std[metric].fillna(0.0).values
        ax.bar(x + (idx - 1) * width, values, width=width, yerr=errors,
               capsize=3, label=metric, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(mean.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"EB Benchmark - {dataset_title}: Accuracy / F1 / ROC-AUC")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / f"{dataset_key}_metrics_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    values = mean["ROC-AUC"].values
    errors = std["ROC-AUC"].fillna(0.0).values
    ax.bar(np.arange(len(mean)), values, yerr=errors, capsize=4, color="#264653", alpha=0.9)
    ax.set_xticks(np.arange(len(mean)))
    ax.set_xticklabels(mean.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"EB Benchmark - {dataset_title}: ROC-AUC Ranking")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    plt.tight_layout()
    fig.savefig(figures_dir / f"{dataset_key}_auc_bars.png", dpi=200)
    plt.close(fig)

    pred_df = pd.DataFrame(raw_predictions)
    if pred_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for algorithm in mean.index:
        algo_df = pred_df[pred_df["Algorithm"] == algorithm]
        if algo_df.empty:
            continue
        y_true = (algo_df["y_true"].to_numpy() == 1).astype(int)
        scores = algo_df["score_margin"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
        except Exception:
            continue
        auc_value = mean.loc[algorithm, "ROC-AUC"]
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{algorithm} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"EB Benchmark - {dataset_title}: ROC Curves")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(figures_dir / f"{dataset_key}_roc_curves.png", dpi=200)
    plt.close(fig)


def save_global_auc_heatmap(all_results: dict[str, list[dict]], output_dir: Path) -> None:
    if not all_results:
        return

    rows = []
    for dataset_key, dataset_results in all_results.items():
        if not dataset_results:
            continue
        frame = summarize_numeric(dataset_results)
        frame.insert(0, "Dataset", dataset_key)
        rows.append(frame[["Dataset", "Algorithm", "ROC-AUC"]])
    if not rows:
        return

    auc_df = pd.concat(rows, ignore_index=True)
    pivot = auc_df.pivot(index="Algorithm", columns="Dataset", values="ROC-AUC")
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)), max(8, 0.4 * len(pivot.index))))
    image = ax.imshow(pivot.fillna(np.nan).to_numpy(), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("EB Benchmark - ROC-AUC Heatmap Across Datasets")
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "global_auc_heatmap.png", dpi=200)
    plt.close(fig)


def save_encoded_circuit_diagram(
    dataset_key: str,
    output_dir: Path,
    circuit_fn,
    n_features: int,
) -> None:
    x1 = np.full(n_features, np.pi / 4)
    x2 = np.full(n_features, np.pi / 3)
    save_circuit_diagram(circuit_fn, x1, x2, f"{dataset_key}_benchmark", str(output_dir / "figures"))


def save_quantum_dataset_circuit_diagrams(dataset_key: str, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    circuits: dict[str, QuantumCircuit] = {}
    if dataset_key == "phase_ising":
        plus = QuantumCircuit(4, name="phase_plus")
        plus.rz(1.0, 0)
        plus.rz(1.0, 1)
        minus = QuantumCircuit(4, name="phase_minus")
        minus.rz(2.2, 2)
        minus.rz(2.2, 3)
        circuits["phase_ising_positive_rotation"] = plus
        circuits["phase_ising_negative_rotation"] = minus
    elif dataset_key == "circuit_outputs":
        qc = QuantumCircuit(4, name="circuit_outputs")
        for qubit in range(4):
            qc.ry(0.8 + 0.2 * qubit, qubit)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        for qubit in range(4):
            qc.rz(0.5 + 0.1 * qubit, qubit)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        circuits["circuit_outputs_example"] = qc
    elif dataset_key == "entanglement":
        pair = QuantumCircuit(4, name="pair_entanglement")
        for qubit in range(4):
            pair.ry(0.6 + 0.2 * qubit, qubit)
        pair.cx(0, 1)
        pair.cx(2, 3)
        pair.rz(0.9, 0)
        pair.rz(1.1, 2)

        chain = QuantumCircuit(4, name="chain_entanglement")
        for qubit in range(4):
            chain.ry(0.6 + 0.2 * qubit, qubit)
        chain.cx(0, 1)
        chain.cx(1, 2)
        chain.cx(2, 3)
        chain.rz(0.9, 1)
        chain.rz(1.1, 2)

        circuits["entanglement_pair_structure"] = pair
        circuits["entanglement_chain_structure"] = chain

    for name, circuit in circuits.items():
        txt_path = figures_dir / f"{name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(circuit.draw(output="text", fold=-1)))
        try:
            fig = circuit.draw(output="mpl", fold=-1)
            fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Could not save circuit diagram for {name}: {exc}")


def choose_train_scope(
    train_scope: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    reps_neg: np.ndarray,
    reps_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if train_scope == "representatives":
        X_scope = np.vstack([reps_neg, reps_pos])
        y_scope = np.array([-1] * len(reps_neg) + [1] * len(reps_pos))
        return X_scope, y_scope, "representatives"
    return X_train, y_train, "full_train"


def evaluate_encoded_dataset(
    dataset_key: str,
    dataset_title: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    circuit_fn,
    output_dir: Path,
    n_partitions: int,
    n_reps: int,
    rep_strategy: str,
    qsvc_train_on: str,
    baseline_train_on: str,
    split_mode: str,
    test_size: float | int,
    train_size: int | None = None,
    scale_to_pi: bool = True,
) -> tuple[list[dict], list[dict]]:
    print("\n" + "=" * 78)
    print(f"BENCHMARK: {dataset_title} [{dataset_key}]")
    print("=" * 78)

    results: list[dict] = []
    raw_predictions: list[dict] = []

    for seed in range(n_partitions):
        t0 = time()
        split_kwargs = {
            "stratify": y_all,
            "random_state": seed,
        }
        if split_mode == "fraction":
            split_kwargs["test_size"] = test_size
        else:
            split_kwargs["train_size"] = train_size
            split_kwargs["test_size"] = test_size

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, **split_kwargs)

        if scale_to_pi:
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        reps_neg = select_representatives(
            X_train[y_train == -1],
            n_reps,
            strategy=rep_strategy,
            random_state=seed,
        )
        reps_pos = select_representatives(
            X_train[y_train == 1],
            n_reps,
            strategy=rep_strategy,
            random_state=seed,
        )
        class_reps = {-1: reps_neg, 1: reps_pos}

        states_neg, _ = build_class_statevectors(reps_neg, circuit_fn)
        states_pos, _ = build_class_statevectors(reps_pos, circuit_fn)

        for agg_mode, label in [
            ("mean", "EB-Mean"),
            ("median", "EB-Median"),
            ("max", "EB-Max"),
        ]:
            y_pred, _, sn, sp = predict_eb(
                X_test,
                reps_neg,
                reps_pos,
                states_neg,
                states_pos,
                circuit_fn,
                agg_mode,
            )
            append_result_rows(
                results,
                raw_predictions,
                dataset_key,
                seed,
                label,
                y_test,
                y_pred,
                sp,
                train_scope="representatives",
                score_neg=sn,
                score_pos=sp,
            )

        gamma = 1.0 / max(1, 2 * X_train.shape[1])
        for similarity, sim_name in [("cosine", "Cosine"), ("rbf", "RBF")]:
            for aggregation, agg_name in [("mean", "Mean"), ("median", "Median"), ("max", "Max")]:
                label = f"CPC-{sim_name}-{agg_name}"
                y_pred, margins, sn, sp = cpc_predict(
                    X_test,
                    class_reps,
                    similarity,
                    aggregation,
                    gamma,
                )
                append_result_rows(
                    results,
                    raw_predictions,
                    dataset_key,
                    seed,
                    label,
                    y_test,
                    y_pred,
                    margins,
                    train_scope="representatives",
                    score_neg=sn,
                    score_pos=sp,
                )

        if qsvc_train_on == "representatives":
            X_q = np.vstack([reps_neg, reps_pos])
            y_q = np.array([-1] * len(reps_neg) + [1] * len(reps_pos))
            qsvc_scope = "representatives"
        else:
            X_q, y_q = X_train, y_train
            qsvc_scope = "full_train"
        qsvc_name = "QSVC (Representatives)" if qsvc_scope == "representatives" else "QSVC (Full Train)"
        y_pred_q, q_scores = run_qsvc(X_q, y_q, X_test)
        append_result_rows(
            results,
            raw_predictions,
            dataset_key,
            seed,
            qsvc_name,
            y_test,
            y_pred_q,
            q_scores,
            train_scope=qsvc_scope,
        )

        X_baseline, y_baseline, baseline_scope = choose_train_scope(
            baseline_train_on,
            X_train,
            y_train,
            reps_neg,
            reps_pos,
        )

        y_pred_nc, nc_scores = nearest_centroid_predict(X_baseline, y_baseline, X_test)
        append_result_rows(
            results,
            raw_predictions,
            dataset_key,
            seed,
            "Nearest Centroid",
            y_test,
            y_pred_nc,
            nc_scores,
            train_scope=baseline_scope,
        )

        for name, estimator in build_common_estimators(X_baseline):
            try:
                estimator.fit(X_baseline, y_baseline)
                y_pred = estimator.predict(X_test)
                scores = estimator_scores(estimator, X_test, positive_label=1)
                append_result_rows(
                    results,
                    raw_predictions,
                    dataset_key,
                    seed,
                    name,
                    y_test,
                    y_pred,
                    scores,
                    train_scope=baseline_scope,
                )
            except Exception as exc:
                print(f"[WARN] {dataset_key} | {name} failed on partition {seed}: {exc}")

        try:
            pairwise_model = LogisticRegression(max_iter=2000, random_state=0)
            pairwise_model = train_pairwise_estimator(pairwise_model, X_baseline, y_baseline, class_reps)
            y_pred_pw, margins_pw, sn_pw, sp_pw = predict_pairwise_estimator(
                pairwise_model,
                X_test,
                class_reps,
                aggregation="max",
            )
            append_result_rows(
                results,
                raw_predictions,
                dataset_key,
                seed,
                "Pairwise LogReg-Max",
                y_test,
                y_pred_pw,
                margins_pw,
                train_scope=baseline_scope,
                score_neg=sn_pw,
                score_pos=sp_pw,
            )
        except Exception as exc:
            print(f"[WARN] {dataset_key} | Pairwise LogReg-Max failed on partition {seed}: {exc}")

        elapsed = (time() - t0) / 60.0
        print(f"  Partition {seed + 1}/{n_partitions} completed in {elapsed:.2f} min")

    prefix = f"benchmark_{dataset_key}"
    save_results(results, raw_predictions, prefix, str(output_dir))
    return results, raw_predictions


def choose_quantum_train_scope(
    train_scope: str,
    class_reps_feats: dict[int, np.ndarray],
    train_feats: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if train_scope == "representatives":
        X_scope = np.vstack([class_reps_feats[-1], class_reps_feats[1]])
        y_scope = np.array([-1] * len(class_reps_feats[-1]) + [1] * len(class_reps_feats[1]))
        return X_scope, y_scope, "representatives"
    return train_feats, train_labels, "full_train"


def evaluate_quantum_dataset(
    dataset_key: str,
    dataset_title: str,
    states: list[np.ndarray],
    labels: np.ndarray,
    output_dir: Path,
    n_splits: int,
    n_reps: int,
    qsvc_train_on: str,
    baseline_train_on: str,
) -> tuple[list[dict], list[dict]]:
    print("\n" + "=" * 78)
    print(f"BENCHMARK: {dataset_title} [{dataset_key}]")
    print("=" * 78)

    results: list[dict] = []
    raw_predictions: list[dict] = []

    features = np.asarray([np.abs(state) ** 2 for state in states])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        t0 = time()
        train_states = [states[i] for i in train_idx]
        test_states = [states[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_feats = features[train_idx]
        test_feats = features[test_idx]

        class_reps_states: dict[int, list[np.ndarray]] = {}
        class_reps_feats: dict[int, np.ndarray] = {}
        for label in (-1, 1):
            idx = np.where(train_labels == label)[0]
            cls_states = [train_states[i] for i in idx]
            rep_idx = quantum_exp.select_representatives(cls_states, n_reps, seed=42 + fold)
            reps_states = [cls_states[i] for i in rep_idx]
            class_reps_states[label] = reps_states
            class_reps_feats[label] = np.asarray([np.abs(state) ** 2 for state in reps_states])

        for aggregation, label in [("mean", "EB-Mean"), ("median", "EB-Median"), ("max", "EB-Max")]:
            preds, score_neg, score_pos = [], [], []
            for state in test_states:
                pred, scores = quantum_exp.eb_classify(state, class_reps_states, aggregation)
                preds.append(pred)
                score_neg.append(scores[-1])
                score_pos.append(scores[1])
            preds_arr = np.asarray(preds, dtype=int)
            score_neg_arr = np.asarray(score_neg, dtype=float)
            score_pos_arr = np.asarray(score_pos, dtype=float)
            append_result_rows(
                results,
                raw_predictions,
                dataset_key,
                fold,
                label,
                test_labels,
                preds_arr,
                score_pos_arr,
                train_scope="representatives",
                score_neg=score_neg_arr,
                score_pos=score_pos_arr,
            )

        gamma = 1.0 / max(1, 2 * features.shape[1])
        for similarity, sim_name in [("cosine", "Cosine"), ("rbf", "RBF")]:
            for aggregation, agg_name in [("mean", "Mean"), ("median", "Median"), ("max", "Max")]:
                label = f"CPC-{sim_name}-{agg_name}"
                y_pred, margins, sn, sp = cpc_predict(
                    test_feats,
                    class_reps_feats,
                    similarity,
                    aggregation,
                    gamma,
                )
                append_result_rows(
                    results,
                    raw_predictions,
                    dataset_key,
                    fold,
                    label,
                    test_labels,
                    y_pred,
                    margins,
                    train_scope="representatives",
                    score_neg=sn,
                    score_pos=sp,
                )

        if qsvc_train_on == "representatives":
            qsvc_train_states = class_reps_states[1] + class_reps_states[-1]
            qsvc_train_labels = np.array([1] * len(class_reps_states[1]) + [-1] * len(class_reps_states[-1]))
            qsvc_scope = "representatives"
        else:
            qsvc_train_states = train_states
            qsvc_train_labels = train_labels
            qsvc_scope = "full_train"
        qsvc_name = "QSVC (Representatives)" if qsvc_scope == "representatives" else "QSVC (Full Train)"
        K_train = quantum_exp.build_fidelity_kernel(qsvc_train_states)
        K_test = quantum_exp.build_fidelity_kernel_test(test_states, qsvc_train_states)
        svc = SVC(kernel="precomputed")
        svc.fit(K_train, qsvc_train_labels)
        preds_q = svc.predict(K_test)
        scores_q = np.asarray(svc.decision_function(K_test), dtype=float)
        append_result_rows(
            results,
            raw_predictions,
            dataset_key,
            fold,
            qsvc_name,
            test_labels,
            preds_q,
            scores_q,
            train_scope=qsvc_scope,
        )

        X_baseline, y_baseline, baseline_scope = choose_quantum_train_scope(
            baseline_train_on,
            class_reps_feats,
            train_feats,
            train_labels,
        )

        y_pred_nc, nc_scores = nearest_centroid_predict(X_baseline, y_baseline, test_feats)
        append_result_rows(
            results,
            raw_predictions,
            dataset_key,
            fold,
            "Nearest Centroid",
            test_labels,
            y_pred_nc,
            nc_scores,
            train_scope=baseline_scope,
        )

        for name, estimator in build_common_estimators(X_baseline):
            try:
                estimator.fit(X_baseline, y_baseline)
                y_pred = estimator.predict(test_feats)
                scores = estimator_scores(estimator, test_feats, positive_label=1)
                append_result_rows(
                    results,
                    raw_predictions,
                    dataset_key,
                    fold,
                    name,
                    test_labels,
                    y_pred,
                    scores,
                    train_scope=baseline_scope,
                )
            except Exception as exc:
                print(f"[WARN] {dataset_key} | {name} failed on fold {fold}: {exc}")

        try:
            pairwise_model = LogisticRegression(max_iter=2000, random_state=0)
            pairwise_model = train_pairwise_estimator(pairwise_model, X_baseline, y_baseline, class_reps_feats)
            y_pred_pw, margins_pw, sn_pw, sp_pw = predict_pairwise_estimator(
                pairwise_model,
                test_feats,
                class_reps_feats,
                aggregation="max",
            )
            append_result_rows(
                results,
                raw_predictions,
                dataset_key,
                fold,
                "Pairwise LogReg-Max",
                test_labels,
                y_pred_pw,
                margins_pw,
                train_scope=baseline_scope,
                score_neg=sn_pw,
                score_pos=sp_pw,
            )
        except Exception as exc:
            print(f"[WARN] {dataset_key} | Pairwise LogReg-Max failed on fold {fold}: {exc}")

        elapsed = (time() - t0) / 60.0
        print(f"  Fold {fold + 1}/{n_splits} completed in {elapsed:.2f} min")

    prefix = f"benchmark_{dataset_key}"
    save_results(results, raw_predictions, prefix, str(output_dir))
    return results, raw_predictions


def run_classical_dataset(dataset_key: str, args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    if dataset_key == "adhoc":
        X_all, y_all = experiment_adhoc.load_dataset(
            n_features=args.adhoc_n_features,
            train_size=args.adhoc_train_size,
            test_size=args.adhoc_test_size,
            gap=args.adhoc_gap,
        )
        total = args.adhoc_train_size + args.adhoc_test_size
        test_fraction = args.adhoc_test_size / total
        dataset_results, raw_predictions = evaluate_encoded_dataset(
            dataset_key="adhoc",
            dataset_title="Ad-hoc synthetic benchmark",
            X_all=X_all,
            y_all=y_all,
            circuit_fn=u_pair_adhoc,
            output_dir=output_dir,
            n_partitions=args.n_partitions,
            n_reps=args.adhoc_n_reps,
            rep_strategy=args.rep_strategy,
            qsvc_train_on=args.qsvc_train_on,
            baseline_train_on=args.classical_baselines_train_on,
            split_mode="fraction",
            test_size=test_fraction,
            scale_to_pi=False,
        )
        if not args.skip_figures:
            save_dataset_comparison_plots("adhoc", "Ad-hoc synthetic benchmark", dataset_results, raw_predictions, output_dir)
            save_encoded_circuit_diagram("adhoc", output_dir, u_pair_adhoc, args.adhoc_n_features)
        return dataset_results, raw_predictions

    if dataset_key == "breastcancer":
        X_all, y_all = experiment_breastcancer.load_dataset(n_features=args.n_features)
        dataset_results, raw_predictions = evaluate_encoded_dataset(
            dataset_key="breastcancer",
            dataset_title="Breast Cancer Wisconsin",
            X_all=X_all,
            y_all=y_all,
            circuit_fn=u_pair_tabular,
            output_dir=output_dir,
            n_partitions=args.n_partitions,
            n_reps=args.n_reps,
            rep_strategy=args.rep_strategy,
            qsvc_train_on=args.qsvc_train_on,
            baseline_train_on=args.classical_baselines_train_on,
            split_mode="fraction",
            test_size=args.test_size,
            scale_to_pi=True,
        )
        if not args.skip_figures:
            save_dataset_comparison_plots("breastcancer", "Breast Cancer Wisconsin", dataset_results, raw_predictions, output_dir)
            save_encoded_circuit_diagram("breastcancer", output_dir, u_pair_tabular, args.n_features)
        return dataset_results, raw_predictions

    if dataset_key == "fraud":
        X_all, y_all = experiment_fraud.load_dataset(
            csv_path=args.fraud_csv,
            n_features=args.n_features,
        )
        dataset_results, raw_predictions = evaluate_encoded_dataset(
            dataset_key="fraud",
            dataset_title="Financial fraud detection",
            X_all=X_all,
            y_all=y_all,
            circuit_fn=u_pair_tabular,
            output_dir=output_dir,
            n_partitions=args.n_partitions,
            n_reps=args.fraud_n_reps,
            rep_strategy=args.rep_strategy,
            qsvc_train_on=args.qsvc_train_on,
            baseline_train_on=args.classical_baselines_train_on,
            split_mode="sizes",
            train_size=args.fraud_train_size,
            test_size=args.fraud_test_size,
            scale_to_pi=True,
        )
        if not args.skip_figures:
            save_dataset_comparison_plots("fraud", "Financial fraud detection", dataset_results, raw_predictions, output_dir)
            save_encoded_circuit_diagram("fraud", output_dir, u_pair_tabular, args.n_features)
        return dataset_results, raw_predictions

    if dataset_key == "iris":
        X_all, y_all = experiment_iris.load_dataset(n_features=args.n_features)
        dataset_results, raw_predictions = evaluate_encoded_dataset(
            dataset_key="iris",
            dataset_title="Iris (setosa vs rest)",
            X_all=X_all,
            y_all=y_all,
            circuit_fn=u_pair_tabular,
            output_dir=output_dir,
            n_partitions=args.n_partitions,
            n_reps=args.n_reps,
            rep_strategy=args.rep_strategy,
            qsvc_train_on=args.qsvc_train_on,
            baseline_train_on=args.classical_baselines_train_on,
            split_mode="fraction",
            test_size=args.test_size,
            scale_to_pi=True,
        )
        if not args.skip_figures:
            save_dataset_comparison_plots("iris", "Iris (setosa vs rest)", dataset_results, raw_predictions, output_dir)
            save_encoded_circuit_diagram("iris", output_dir, u_pair_tabular, args.n_features)
        return dataset_results, raw_predictions

    if dataset_key == "wine":
        X_all, y_all = experiment_wine.load_dataset(n_features=args.n_features)
        dataset_results, raw_predictions = evaluate_encoded_dataset(
            dataset_key="wine",
            dataset_title="Wine (class 0 vs rest)",
            X_all=X_all,
            y_all=y_all,
            circuit_fn=u_pair_tabular,
            output_dir=output_dir,
            n_partitions=args.n_partitions,
            n_reps=args.n_reps,
            rep_strategy=args.rep_strategy,
            qsvc_train_on=args.qsvc_train_on,
            baseline_train_on=args.classical_baselines_train_on,
            split_mode="fraction",
            test_size=args.test_size,
            scale_to_pi=True,
        )
        if not args.skip_figures:
            save_dataset_comparison_plots("wine", "Wine (class 0 vs rest)", dataset_results, raw_predictions, output_dir)
            save_encoded_circuit_diagram("wine", output_dir, u_pair_tabular, args.n_features)
        return dataset_results, raw_predictions

    raise ValueError(f"Unknown classical dataset: {dataset_key}")


def run_quantum_dataset(dataset_key: str, args, output_dir: Path) -> tuple[list[dict], list[dict]]:
    if dataset_key == "phase_ising":
        states, labels = quantum_exp.generate_phase_sensitive_ising(
            args.quantum_n_qubits,
            args.quantum_samples_per_class,
            seed=args.quantum_seed,
        )
        title = "Phase-Rotated Ising"
    elif dataset_key == "circuit_outputs":
        states, labels = quantum_exp.generate_circuit_output_dataset(
            args.quantum_n_qubits,
            args.quantum_samples_per_class,
            seed=args.quantum_seed,
        )
        title = "Circuit Outputs"
    elif dataset_key == "entanglement":
        states, labels = quantum_exp.generate_entanglement_dataset(
            args.quantum_n_qubits,
            args.quantum_samples_per_class,
            seed=args.quantum_seed,
        )
        title = "Entanglement Classes"
    else:
        raise ValueError(f"Unknown quantum dataset: {dataset_key}")

    dataset_results, raw_predictions = evaluate_quantum_dataset(
        dataset_key=dataset_key,
        dataset_title=title,
        states=states,
        labels=labels,
        output_dir=output_dir,
        n_splits=args.quantum_n_splits,
        n_reps=args.quantum_n_reps,
        qsvc_train_on=args.qsvc_train_on,
        baseline_train_on=args.classical_baselines_train_on,
    )
    if not args.skip_figures:
        save_dataset_comparison_plots(dataset_key, title, dataset_results, raw_predictions, output_dir)
        save_quantum_dataset_circuit_diagrams(dataset_key, output_dir)
    return dataset_results, raw_predictions


def normalize_datasets(requested: list[str]) -> list[str]:
    expanded = []
    for item in requested:
        if item == "all":
            expanded.extend(ALL_DATASETS)
        elif item == "classical":
            expanded.extend(CLASSICAL_DATASETS)
        elif item == "quantum":
            expanded.extend(QUANTUM_DATASETS)
        else:
            expanded.append(item)
    ordered = []
    for name in expanded:
        if name not in ordered:
            ordered.append(name)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified benchmark for EB, CPC, QSVC and extended classical baselines."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=ALL_DATASETS + ["all", "classical", "quantum"],
        help="Datasets to run. Use 'all', 'classical', or 'quantum' shortcuts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where benchmark outputs will be written.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip generating comparative plots and circuit images.",
    )

    parser.add_argument("--n-partitions", type=int, default=10)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--rep-strategy", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument(
        "--qsvc-train-on",
        choices=["representatives", "full_train"],
        default="representatives",
        help="Train QSVC on representative subset or full train split.",
    )
    parser.add_argument(
        "--classical-baselines-train-on",
        choices=["representatives", "full_train"],
        default="full_train",
        help="Train standard classical baselines on representatives or full train split.",
    )

    parser.add_argument("--adhoc-n-features", type=int, default=2)
    parser.add_argument("--adhoc-train-size", type=int, default=500)
    parser.add_argument("--adhoc-test-size", type=int, default=150)
    parser.add_argument("--adhoc-gap", type=float, default=0.3)
    parser.add_argument("--adhoc-n-reps", type=int, default=20)

    parser.add_argument("--fraud-csv", type=str, default="synthetic_fraud_dataset.csv")
    parser.add_argument("--fraud-train-size", type=int, default=4000)
    parser.add_argument("--fraud-test-size", type=int, default=1000)
    parser.add_argument("--fraud-n-reps", type=int, default=30)

    parser.add_argument("--quantum-n-qubits", type=int, default=4)
    parser.add_argument("--quantum-samples-per-class", type=int, default=60)
    parser.add_argument("--quantum-n-reps", type=int, default=6)
    parser.add_argument("--quantum-n-splits", type=int, default=10)
    parser.add_argument("--quantum-seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    datasets = normalize_datasets(args.datasets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    print("\n" + "=" * 78)
    print("UNIFIED BENCHMARK: EB / CPC / QSVC / CLASSICAL BASELINES")
    print("=" * 78)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output dir: {output_dir}")
    print(f"QSVC train scope: {args.qsvc_train_on}")
    print(f"Classical baseline scope: {args.classical_baselines_train_on}")

    for dataset_key in datasets:
        if dataset_key == "fraud" and not os.path.exists(args.fraud_csv):
            print(f"[SKIP] fraud dataset: CSV not found at {args.fraud_csv}")
            continue

        if dataset_key in CLASSICAL_DATASETS:
            dataset_results, _ = run_classical_dataset(dataset_key, args, output_dir)
        else:
            dataset_results, _ = run_quantum_dataset(dataset_key, args, output_dir)
        all_results[dataset_key] = dataset_results

    save_global_outputs(all_results, output_dir, vars(args))
    if not args.skip_figures:
        save_global_auc_heatmap(all_results, output_dir)
    print("\nBenchmark outputs saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()

