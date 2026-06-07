"""
make_cpc_vs_eb_figure.py
═══════════════════════════════════════════════════════════════════════════
Regenerate the EB-vs-CPC comparison figure (`cpc_vs_eb_bars.png`) used in the
thesis, spanning ALL eight datasets of the TFM — the five classical ones AND
the three quantum-native ones (phase Ising, circuit outputs, entanglement).

Numbers are read from the benchmark source-of-truth CSV
(`outputs/benchmark_all_methods/benchmark_all_datasets_summary_numeric.csv`),
so the figure stays consistent with the tables in the manuscript. For each
dataset it plots the best EB variant vs the best CPC variant in ROC-AUC.

Output: outputs/figures/cpc_vs_eb_bars.png (+ .pdf)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = os.path.join("outputs", "benchmark_all_methods",
                   "benchmark_all_datasets_summary_numeric.csv")
FIG_DIR = os.path.join("outputs", "figures")

# dataset key -> (display name, is_quantum)
DATASETS = [
    ("adhoc",           "Ad-hoc",        False),
    ("breastcancer",    "Cáncer mama",   False),
    ("fraud",           "Fraude",        False),
    ("iris",            "Iris",          False),
    ("wine",            "Wine",          False),
    ("phase_ising",     "Ising fase",    True),
    ("circuit_outputs", "Salidas circ.", True),
    ("entanglement",    "Entrelaz.",     True),
]

EB_COLOR  = "#1D5A4E"   # verde (cuántico, fidelidad)
CPC_COLOR = "#C07A2B"   # naranja (clásico, prototipo)


def best_of(df, prefix, metric="ROC-AUC"):
    sub = df[df.Algorithm.str.startswith(prefix)]
    row = sub.loc[sub[metric].idxmax()]
    return float(row[metric]), str(row.Algorithm)


def main():
    df = pd.read_csv(CSV)
    eb, cpc, deltas, labels, quantum = [], [], [], [], []
    print(f"{'dataset':16s} {'best EB':>8s} {'best CPC':>8s} {'delta':>7s}")
    for key, name, is_q in DATASETS:
        g = df[df.Dataset == key]
        e, e_name = best_of(g, "EB")
        c, c_name = best_of(g, "CPC")
        eb.append(e); cpc.append(c); deltas.append(e - c)
        labels.append(name); quantum.append(is_q)
        print(f"{key:16s} {e:8.3f} {c:8.3f} {e-c:+7.3f}   "
              f"(EB={e_name}, CPC={c_name})")

    x = np.arange(len(DATASETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    # shade the quantum-dataset region
    first_q = next(i for i, q in enumerate(quantum) if q)
    ax.axvspan(first_q - 0.5, len(x) - 0.5, color="#EAF2EE", zorder=0)

    b1 = ax.bar(x - w/2, eb,  w, label="Mejor EB (cuántico)",
                color=EB_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w/2, cpc, w, label="Mejor CPC (prototipo clásico)",
                color=CPC_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

    ymin = 0.5
    ax.set_ylim(ymin, 1.06)
    # delta labels above each pair
    for xi, e, c, d in zip(x, eb, cpc, deltas):
        top = max(e, c)
        col = EB_COLOR if d > 1e-9 else (CPC_COLOR if d < -1e-9 else "#666666")
        ax.text(xi, top + 0.012, f"$\\Delta$={d:+.3f}", ha="center",
                va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("EB frente a CPC en los ocho conjuntos del TFM (ROC-AUC)",
                 fontsize=12.5)

    # group annotations
    ax.text((first_q - 1) / 2, 1.045, "Conjuntos clásicos", ha="center",
            fontsize=9, style="italic", color="#5B6963")
    ax.text((first_q + len(x) - 1) / 2, 1.045, "Conjuntos cuántico-nativos",
            ha="center", fontsize=9, style="italic", color="#27604F")

    ax.legend(loc="lower left", frameon=True, fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, "cpc_vs_eb_bars.png")
    pdf = os.path.join(FIG_DIR, "cpc_vs_eb_bars.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"\nSaved: {png}\nSaved: {pdf}")


if __name__ == "__main__":
    main()
