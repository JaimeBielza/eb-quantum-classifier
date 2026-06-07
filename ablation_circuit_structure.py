"""
ablation_circuit_structure.py
═══════════════════════════════════════════════════════════════════════════
Ablation study for Section 3.7.3 of the TFM:
"Does the inter-register circuit structure contribute to EB's performance?"

Three U_pair variants are compared on the Breast Cancer Wisconsin dataset
using the *identical* EB pipeline (PCA→4 features, RY angle encoding,
MinMax scaling to [0, π], 10 stratified partitions, kmeans representatives):

  - EB-Entangled : the original circuit  (RY encoding + CNOT inter-register)
  - EB-Product   : RY encoding only, NO inter-register gates  (product state)
  - EB-Hadamard  : H on every qubit BEFORE the RY encoding, then CNOT

Mathematical expectation: the inter-register layer is the SAME fixed unitary
U applied to *every* statevector (both test and class states). In the
fidelity |<a|U†U|b>|² = |<a|b>|² it cancels (U†U = I), so all variants must
produce identical fidelities — hence identical metrics on every partition.
This script confirms that empirically and writes a table + figure.

Outputs
───────
  outputs/ablation_circuit_structure/ablation_summary.csv
  outputs/ablation_circuit_structure/ablation_raw.csv
  outputs/figures/ablation_circuit.pdf   (grouped-bar figure)
  outputs/ablation_circuit_structure/ablation_circuit.png  (preview)
"""

from __future__ import annotations

import os
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from eb_shared import (
    build_class_statevectors,
    compute_metrics,
    predict_eb,
    select_representatives,
)

OUT_DIR  = os.path.join("outputs", "ablation_circuit_structure")
FIG_DIR  = os.path.join("outputs", "figures")
METRICS  = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
AGG_MODES = ["mean", "median", "max"]


# ─────────────────────────────────────────────────────────────────────────
# The three circuit variants (all share the RY angle encoding of u_pair_tabular)
# ─────────────────────────────────────────────────────────────────────────
def u_pair_entangled(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """Original EB tabular circuit: RY encoding + CNOT inter-register."""
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


def u_pair_product(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """RY encoding only — NO inter-register gates (pure product state)."""
    n  = len(x1)
    qc = QuantumCircuit(2 * n)
    for i, v in enumerate(x1):
        qc.ry(float(v), i)
    for i, v in enumerate(x2):
        qc.ry(float(v), n + i)
    return qc


def u_pair_hadamard(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """Alternative entangling structure: H on every qubit before encoding, then CNOT."""
    n  = len(x1)
    qc = QuantumCircuit(2 * n)
    for q in range(2 * n):
        qc.h(q)
    for i, v in enumerate(x1):
        qc.ry(float(v), i)
    for i, v in enumerate(x2):
        qc.ry(float(v), n + i)
    qc.barrier()
    for i in range(n):
        qc.cx(i, n + i)
    return qc


VARIANTS = [
    ("EB-Entangled", u_pair_entangled),
    ("EB-Product",   u_pair_product),
    ("EB-Hadamard",  u_pair_hadamard),
]


# ─────────────────────────────────────────────────────────────────────────
def load_dataset(n_features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data.astype(float))
    X = PCA(n_components=n_features, random_state=0).fit_transform(X)
    y = np.where(data.target == 0, -1, 1)         # malignant -1, benign +1
    return X, y


def run(n_partitions=10, n_features=4, test_size=0.30,
        n_representatives=20, rep_strategy="kmeans") -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 74)
    print("ABLATION — circuit structure (Breast Cancer)")
    print(f"  variants={[v[0] for v in VARIANTS]}")
    print(f"  n_features={n_features}  n_reps={n_representatives}"
          f"  partitions={n_partitions}  strategy={rep_strategy}")
    print("=" * 74)

    X_all, y_all = load_dataset(n_features=n_features)
    rows = []

    for seed in range(n_partitions):
        t0 = time()
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size,
            stratify=y_all, random_state=seed)
        scaler  = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Representatives are chosen ONCE per partition (independent of circuit).
        reps_neg = select_representatives(
            X_train[y_train == -1], n_representatives,
            strategy=rep_strategy, random_state=seed)
        reps_pos = select_representatives(
            X_train[y_train ==  1], n_representatives,
            strategy=rep_strategy, random_state=seed)

        for vname, fn in VARIANTS:
            states_neg, _ = build_class_statevectors(reps_neg, fn)
            states_pos, _ = build_class_statevectors(reps_pos, fn)
            for agg in AGG_MODES:
                y_pred, _, sn, sp = predict_eb(
                    X_test, reps_neg, reps_pos,
                    states_neg, states_pos, fn, agg)
                m = compute_metrics(y_test, y_pred, scores=sp)
                m.update({"Variant": vname, "Aggregation": agg, "Partition": seed})
                rows.append(m)
        print(f"  partition {seed+1}/{n_partitions} done in {(time()-t0)/60:.2f} min")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT_DIR, "ablation_raw.csv"), index=False)

    # ── Exact-identity check across variants (per partition & aggregation) ──
    print("\nMaximum metric spread across the three circuit variants")
    print("(per partition & aggregation — should be ~0):")
    max_spread = 0.0
    for (agg, part), g in raw.groupby(["Aggregation", "Partition"]):
        for col in METRICS:
            spread = g[col].max() - g[col].min()
            max_spread = max(max_spread, abs(spread))
    print(f"  max |spread| over all metrics/partitions/aggregations = {max_spread:.2e}")
    identical = max_spread < 1e-9
    print(f"  -> variants are {'IDENTICAL' if identical else 'NOT identical'} "
          f"(threshold 1e-9)\n")

    # ── Summary table (mean ± std over partitions, for EB-Mean) ──
    summary = (raw[raw.Aggregation == "mean"]
               .groupby("Variant")[METRICS]
               .agg(["mean", "std"]))
    summary.to_csv(os.path.join(OUT_DIR, "ablation_summary.csv"))
    print("Summary (EB-Mean, mean over partitions):")
    print(summary.xs("mean", axis=1, level=1).round(4).to_string())

    _make_figure(raw)
    print(f"\nSaved: {OUT_DIR}/ablation_summary.csv, ablation_raw.csv")
    print(f"Saved: {FIG_DIR}/ablation_circuit.pdf  (+ preview png)")


def _make_figure(raw: pd.DataFrame) -> None:
    """Grouped bars: three circuit variants × five metrics (EB-Mean, averaged)."""
    sub = raw[raw.Aggregation == "mean"].groupby("Variant")[METRICS].mean()
    sub = sub.reindex([v[0] for v in VARIANTS])

    colors = {"EB-Entangled": "#1D5A4E",
              "EB-Product":   "#7BA494",
              "EB-Hadamard":  "#C8A24B"}
    x = np.arange(len(METRICS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for k, vname in enumerate(sub.index):
        ax.bar(x + (k - 1) * w, sub.loc[vname].values, w,
               label=vname, color=colors[vname],
               edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor de la métrica", fontsize=10)
    ax.set_title("Ablación de la estructura del circuito (Breast Cancer, EB-Mean)\n"
                 "Las tres variantes producen resultados idénticos",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_circuit.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "ablation_circuit.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-partitions", type=int, default=10)
    p.add_argument("--n-reps", type=int, default=20)
    args = p.parse_args()
    run(n_partitions=args.n_partitions, n_representatives=args.n_reps)
