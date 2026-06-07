"""
experiment_fraud.py
EB paper — Financial fraud detection experiment.

4 features (log-transformed monetary features), RY angle encoding,
4 qubits per register.

Outputs (in outputs/)

  fraud_raw_results.csv
  fraud_raw_predictions.csv
  fraud_summary.csv
  fraud_summary_numeric.csv
  fraud_accuracy.png
  fraud_f1_score.png
  fraud_circuit.png / .txt
"""

from __future__ import annotations

import os
from time import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from eb_shared import (
    CLASSICAL_MODELS,
    build_class_statevectors,
    compute_metrics,
    predict_eb,
    run_qsvc,
    save_bar_plots,
    save_circuit_diagram,
    save_results,
    select_representatives,
    u_pair_tabular,
)

OUTPUT_DIR    = "outputs"
CIRCUIT_FN    = u_pair_tabular
DATASET_NAME  = "fraud"
DATASET_TITLE = "Financial fraud detection"



def load_dataset(
    csv_path:   str = "synthetic_fraud_dataset.csv",
    n_features: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the fraud dataset.
    Log-transform monetary features; normalize to [0,Ï€] inside the loop
    (fit on train only) to avoid leakage.
    """
    df = pd.read_csv(csv_path)
    df["Transaction_Amount"] = np.log1p(df["Transaction_Amount"])
    df["Account_Balance"]    = np.log1p(df["Account_Balance"])

    feature_cols = [
        "Transaction_Amount",
        "Account_Balance",
        "Daily_Transaction_Count",
        "Risk_Score",
    ]
    X = df[feature_cols].astype(float).values
    y = df["Fraud_Label"].replace({0: -1, 1: 1}).astype(int).values
    return X, y



def main(
    csv_path:          str   = "synthetic_fraud_dataset.csv",
    n_partitions:      int   = 10,
    n_features:        int   = 4,
    train_size:        int   = 4000,   # dataset has ~50k rows; 
    test_size:         int   = 1000,   # robust test set
    n_representatives: int   = 30,     # more reps â†’ better class coverage
    rep_strategy:      str   = "kmeans",
    qsvc_train_on:     str   = "representatives",
    output_dir:        str   = OUTPUT_DIR,
    # legacy fraud-specific args (ignored in unified pipeline)
    kmeans_n_clusters:  int | None = None,
    kmeans_n_per_cluster: int | None = None,
    **_unused,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print("FRAUD EB vs QSVC vs CLASSICAL BASELINES")
    print("=" * 78)
    print(f"  n_features={n_features}  n_reps={n_representatives}"
          f"  strategy={rep_strategy}  qsvc_train_on={qsvc_train_on}")

    X_all, y_all = load_dataset(csv_path=csv_path, n_features=n_features)
    results, raw_predictions = [], []

    for seed in range(n_partitions):
        t0 = time()
        print(f"\n--- Partition {seed+1}/{n_partitions} | seed={seed} ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            train_size=train_size, test_size=test_size,
            stratify=y_all, random_state=seed,
        )

        scaler  = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        reps_neg = select_representatives(
            X_train[y_train == -1], n_representatives,
            strategy=rep_strategy, random_state=seed)
        reps_pos = select_representatives(
            X_train[y_train ==  1], n_representatives,
            strategy=rep_strategy, random_state=seed)

        states_neg, neg_pairs = build_class_statevectors(reps_neg, CIRCUIT_FN)
        states_pos, pos_pairs = build_class_statevectors(reps_pos, CIRCUIT_FN)

        
        for agg_mode, label in [("mean",   "EB-Mean"),
                                 ("median", "EB-Median"),
                                 ("max",    "EB-Max")]:
            y_pred, margins, sn, sp = predict_eb(
                X_test, reps_neg, reps_pos,
                states_neg, states_pos, CIRCUIT_FN, agg_mode,
            )
            m = compute_metrics(y_test, y_pred, scores=sp)
            m.update({"Algorithm": label, "Partition": seed,
                      "Rep-Strategy": rep_strategy,
                      "n_reps_neg": len(reps_neg),
                      "n_reps_pos": len(reps_pos),
                      "n_class_states_neg": len(neg_pairs),
                      "n_class_states_pos": len(pos_pairs)})
            results.append(m)
            raw_predictions.extend(
                {"Partition": seed, "Algorithm": label,
                 "y_true": int(yt), "y_pred": int(yp),
                 "score_margin": float(mg),
                 "score_neg": float(sn_), "score_pos": float(sp_)}
                for yt, yp, mg, sn_, sp_ in zip(
                    y_test, y_pred, margins, sn, sp)
            )

        
        if qsvc_train_on == "representatives":
            X_q = np.vstack([reps_neg, reps_pos])
            y_q = np.array([-1]*len(reps_neg) + [1]*len(reps_pos))
        else:
            X_q, y_q = X_train, y_train

        y_pred_q, q_scores = run_qsvc(X_q, y_q, X_test)
        m = compute_metrics(y_test, y_pred_q, scores=q_scores)
        m.update({"Algorithm": "QSVC (ZZ Feature Map)", "Partition": seed,
                  "Rep-Strategy": rep_strategy,
                  "n_reps_neg": len(reps_neg), "n_reps_pos": len(reps_pos),
                  "n_class_states_neg": len(neg_pairs),
                  "n_class_states_pos": len(pos_pairs)})
        results.append(m)
        raw_predictions.extend(
            {"Partition": seed, "Algorithm": "QSVC (ZZ Feature Map)",
             "y_true": int(yt), "y_pred": int(yp),
             "score_margin": float(sc),
             "score_neg": float("nan"), "score_pos": float("nan")}
            for yt, yp, sc in zip(y_test, y_pred_q, q_scores)
        )

        
        X_reps = np.vstack([reps_neg, reps_pos])
        y_reps = np.array([-1]*len(reps_neg) + [1]*len(reps_pos))

        for name, model in CLASSICAL_MODELS:
            clf    = clone(model)
            clf.fit(X_reps, y_reps)
            y_pred = clf.predict(X_test)
            scores = (clf.predict_proba(X_test)[:, 1]
                      if hasattr(clf, "predict_proba")
                      else clf.decision_function(X_test)
                      if hasattr(clf, "decision_function")
                      else y_pred.astype(float))
            m = compute_metrics(y_test, y_pred, scores=scores)
            m.update({"Algorithm": name, "Partition": seed,
                      "Rep-Strategy": rep_strategy,
                      "n_reps_neg": len(reps_neg),
                      "n_reps_pos": len(reps_pos),
                      "n_class_states_neg": len(neg_pairs),
                      "n_class_states_pos": len(pos_pairs)})
            results.append(m)
            raw_predictions.extend(
                {"Partition": seed, "Algorithm": name,
                 "y_true": int(yt), "y_pred": int(yp),
                 "score_margin": float(sc),
                 "score_neg": float("nan"), "score_pos": float("nan")}
                for yt, yp, sc in zip(y_test, y_pred, scores)
            )

        print(f"  done in {(time()-t0)/60:.2f} min")

    save_results(results, raw_predictions, DATASET_NAME, output_dir)
    save_bar_plots(results, DATASET_NAME, output_dir, DATASET_TITLE)
    save_circuit_diagram(
        CIRCUIT_FN,
        np.full(n_features, np.pi / 4),
        np.full(n_features, np.pi / 3),
        DATASET_NAME, output_dir,
    )


if __name__ == "__main__":
    main()
