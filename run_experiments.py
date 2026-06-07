"""
run_experiments.py
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
Unified experiment runner for the EB paper.

Runs any combination of the five datasets and generates all paper outputs.

STRUCTURE
â”€â”€â”€â”€â”€â”€â”€â”€â”€
  run_experiments.py          â† this file (main entry point)
  eb_shared.py               â† shared utilities (circuits, metrics, saving)
  experiment_adhoc.py         â† ad-hoc synthetic benchmark
  experiment_breastcancer.py  â† Wisconsin Diagnostic Breast Cancer
  experiment_fraud.py         â† financial fraud detection
  experiment_iris.py          â† Iris (setosa vs rest)
  experiment_wine.py          â† Wine (class 0 vs rest)
  outputs/                    â† all CSV and PNG outputs land here

USAGE
â”€â”€â”€â”€â”€
  # Run all five datasets with defaults
  python run_experiments.py

  # Run specific datasets
  python run_experiments.py --datasets iris wine

  # Run with custom settings
  python run_experiments.py --datasets adhoc breastcancer \\
      --n-partitions 10 --n-reps 20 --rep-strategy kmeans

  # Run without generating summary figures at the end
  python run_experiments.py --skip-figures

  # Run fraud with its CSV in a non-default location
  python run_experiments.py --datasets fraud \\
      --fraud-csv /path/to/synthetic_fraud_dataset.csv

AVAILABLE DATASETS
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  adhoc          Synthetic quantum benchmark (2 features, ZZFeatureMap circuit)
  breastcancer   Wisconsin Diagnostic Breast Cancer (4 features via PCA)
  fraud          Semi-synthetic financial fraud (4 features, requires CSV)
  iris           Iris setosa vs rest (4 features via PCA)
  wine           Wine class 0 vs rest (4 features via PCA)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import time

# â”€â”€ Path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Make sure eb_shared and the experiment scripts are importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Dataset registry
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

DATASET_SCRIPTS: dict[str, str] = {
    "adhoc":         "experiment_adhoc.py",
    "breastcancer":  "experiment_breastcancer.py",
    "fraud":         "experiment_fraud.py",
    "iris":          "experiment_iris.py",
    "wine":          "experiment_wine.py",
}

ALL_DATASETS = list(DATASET_SCRIPTS.keys())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Module loader
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Per-dataset runners
# Each function translates the unified CLI args into the dataset's main() call
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _run_adhoc(args: argparse.Namespace) -> None:
    mod = _load_module("experiment_adhoc",
                       BASE_DIR / DATASET_SCRIPTS["adhoc"])
    mod.main(
        n_partitions=args.n_partitions,
        n_features=args.adhoc_n_features,
        train_size=args.adhoc_train_size,
        test_size=args.adhoc_test_size,
        gap=args.adhoc_gap,
        n_representatives=args.adhoc_n_reps,
        rep_strategy=args.rep_strategy,
        qsvc_train_on=args.qsvc_train_on,
        output_dir=str(OUTPUT_DIR),
    )


def _run_breastcancer(args: argparse.Namespace) -> None:
    mod = _load_module("experiment_breastcancer",
                       BASE_DIR / DATASET_SCRIPTS["breastcancer"])
    mod.main(
        n_partitions=args.n_partitions,
        n_features=args.n_features,
        test_size=args.test_size,
        n_representatives=args.n_reps,
        rep_strategy=args.rep_strategy,
        qsvc_train_on=args.qsvc_train_on,
        output_dir=str(OUTPUT_DIR),
    )


def _run_fraud(args: argparse.Namespace) -> None:
    if not os.path.exists(args.fraud_csv):
        print(f"[SKIP] fraud: CSV not found at '{args.fraud_csv}'")
        return
    mod = _load_module("experiment_fraud",
                       BASE_DIR / DATASET_SCRIPTS["fraud"])
    mod.main(
        csv_path=args.fraud_csv,
        n_partitions=args.n_partitions,
        train_size=args.fraud_train_size,
        test_size=args.fraud_test_size,
        n_representatives=args.fraud_n_reps,
        rep_strategy=args.rep_strategy,
        qsvc_train_on=args.qsvc_train_on,
        output_dir=str(OUTPUT_DIR),
    )


def _run_iris(args: argparse.Namespace) -> None:
    mod = _load_module("experiment_iris",
                       BASE_DIR / DATASET_SCRIPTS["iris"])
    mod.main(
        n_partitions=args.n_partitions,
        n_features=args.n_features,
        test_size=args.test_size,
        n_representatives=args.n_reps,
        rep_strategy=args.rep_strategy,
        qsvc_train_on=args.qsvc_train_on,
        output_dir=str(OUTPUT_DIR),
    )


def _run_wine(args: argparse.Namespace) -> None:
    mod = _load_module("experiment_wine",
                       BASE_DIR / DATASET_SCRIPTS["wine"])
    mod.main(
        n_partitions=args.n_partitions,
        n_features=args.n_features,
        test_size=args.test_size,
        n_representatives=args.n_reps,
        rep_strategy=args.rep_strategy,
        qsvc_train_on=args.qsvc_train_on,
        output_dir=str(OUTPUT_DIR),
    )


RUNNERS = {
    "adhoc":        _run_adhoc,
    "breastcancer": _run_breastcancer,
    "fraud":        _run_fraud,
    "iris":         _run_iris,
    "wine":         _run_wine,
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EB paper â€” unified experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # â”€â”€ Dataset selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Which datasets to run.",
    )

    # â”€â”€ Shared settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument("--n-partitions", type=int, default=10,
                   help="Number of stratified train/test splits.")
    p.add_argument("--n-features",   type=int, default=4,
                   help="Feature dimension for tabular datasets.")
    p.add_argument("--test-size",    type=float, default=0.30,
                   help="Test fraction for tabular datasets.")
    p.add_argument("--n-reps",       type=int, default=20,
                   help="Representatives per class (tabular datasets).")
    p.add_argument(
        "--rep-strategy", choices=["kmeans", "random"], default="kmeans",
        help="Representative selection strategy.",
    )
    p.add_argument(
        "--qsvc-train-on",
        choices=["representatives", "full_train"],
        default="representatives",
        help="Train QSVC on representative subset or full training set.",
    )
    p.add_argument(
        "--skip-figures", action="store_true",
        help="Skip generating summary figures after experiments.",
    )

    # â”€â”€ Ad-hoc specific â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    adhoc = p.add_argument_group("ad-hoc dataset")
    adhoc.add_argument("--adhoc-n-features",  type=int,   default=2)
    adhoc.add_argument("--adhoc-train-size",  type=int,   default=500)
    adhoc.add_argument("--adhoc-test-size",   type=int,   default=150)
    adhoc.add_argument("--adhoc-gap",         type=float, default=0.3)
    adhoc.add_argument("--adhoc-n-reps",      type=int,   default=20)

    # â”€â”€ Fraud specific â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fraud = p.add_argument_group("fraud dataset")
    fraud.add_argument(
        "--fraud-csv", type=str, default="synthetic_fraud_dataset.csv",
        help="Path to the fraud CSV file.",
    )
    fraud.add_argument("--fraud-train-size", type=int, default=400)
    fraud.add_argument("--fraud-test-size",  type=int, default=250)
    fraud.add_argument("--fraud-n-reps",     type=int, default=20)

    return p


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Main
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    os.chdir(BASE_DIR)

    print("\n" + "â•”" + "â•" * 76 + "â•—")
    print("â•‘  EB EXPERIMENT RUNNER" + " " * 52 + "â•‘")
    print("â• " + "â•" * 76 + "â•£")
    print(f"â•‘  Datasets     : {', '.join(args.datasets)}")
    print(f"â•‘  Partitions   : {args.n_partitions}")
    print(f"â•‘  Reps/class   : {args.n_reps}  (tabular)  |  "
          f"{args.adhoc_n_reps}  (ad-hoc)")
    print(f"â•‘  Rep strategy : {args.rep_strategy}")
    print(f"â•‘  QSVC trains on: {args.qsvc_train_on}")
    print(f"â•‘  Output dir   : {OUTPUT_DIR}")
    print("â•š" + "â•" * 76 + "â•\n")

    total_start = time()
    completed, failed = [], []

    for ds in args.datasets:
        print(f"\n{'â”'*78}")
        print(f"  STARTING: {ds.upper()}")
        print(f"{'â”'*78}")
        t0 = time()
        try:
            RUNNERS[ds](args)
            elapsed = (time() - t0) / 60
            completed.append(ds)
            print(f"\n  âœ“ {ds} completed in {elapsed:.1f} min")
        except Exception as exc:
            failed.append(ds)
            print(f"\n  âœ— {ds} FAILED: {exc}")
            import traceback
            traceback.print_exc()

    # â”€â”€ Final report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_min = (time() - total_start) / 60
    print("\n" + "â•" * 78)
    print("EXPERIMENT SUMMARY")
    print("â•" * 78)
    print(f"  Completed : {completed}")
    print(f"  Failed    : {failed}")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Outputs   : {OUTPUT_DIR}/")
    print()

    if completed:
        print("Files generated per dataset:")
        for ds in completed:
            for suffix in ["_summary.csv", "_raw_results.csv",
                           "_raw_predictions.csv", "_summary_numeric.csv",
                           "_accuracy.png", "_f1_score.png"]:
                path = OUTPUT_DIR / f"{ds}{suffix}"
                status = "âœ“" if path.exists() else "Â·"
                print(f"    {status}  {path.name}")
        print()

    if failed:
        print("To retry failed datasets:")
        print(f"  python run_experiments.py --datasets {' '.join(failed)}")
        print()


if __name__ == "__main__":
    main()

