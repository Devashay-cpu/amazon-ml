"""
Phase 4 — Baseline match classifier

Trains and evaluates a simple, interpretable logistic-regression baseline
on the Phase 3 candidate-pair feature dataset
(features/train_pair_features.csv), to establish how well the Phase 3
engineered features separate matches from non-matches.

This is intentionally NOT an advanced model: Logistic Regression with
class_weight="balanced" inside a small reproducible sklearn Pipeline
(sentinel-aware imputation -> scaling -> classifier).

Usage:
    python src/train_baseline.py
    python src/train_baseline.py --input features/train_pair_features.csv --test-size 0.2 --seed 42

Writes:
    reports/phase4_baseline_report.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, MISSING

ID_COLUMNS = ["s1_entity_id", "candidate_entity_id", "source"]
LABEL_COLUMN = "label"
RANDOM_SEED_DEFAULT = 42


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent
    if (candidate / "dataset").exists():
        return candidate
    for parent in script_path.parents:
        if (parent / "dataset").exists():
            return parent
    return Path.cwd()


def load_dataset(input_path: Path) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Loads the Phase 3 feature CSV, validates schema against features.FEATURE_NAMES
    (reused, not redefined), and reports data-quality stats. Rows with a missing
    or unparseable label are dropped -- explicitly counted and reported, never
    silently discarded.
    """
    df = pd.read_csv(input_path)

    missing_id_cols = [c for c in ID_COLUMNS if c not in df.columns]
    if missing_id_cols:
        raise ValueError(f"Expected identifier columns not found in {input_path}: {missing_id_cols}")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COLUMN}' not found in {input_path}")

    feature_columns = [c for c in df.columns if c not in ID_COLUMNS + [LABEL_COLUMN]]
    schema_mismatch = sorted(set(feature_columns) ^ set(FEATURE_NAMES))
    if schema_mismatch:
        print(f"[train_baseline] WARNING: feature columns in {input_path} differ from the "
              f"canonical Phase 3 schema (features.FEATURE_NAMES). Differing columns: "
              f"{schema_mismatch}. Proceeding with the columns actually found in the file, "
              f"reusing (not redefining) whichever of the canonical features are present.")
    # Only ever use columns that are BOTH present in the file AND part of the
    # canonical Phase 3 feature schema -- never an identifier, never an
    # unexpected extra column.
    feature_columns = [c for c in FEATURE_NAMES if c in df.columns]

    n_rows_before = len(df)
    label_numeric = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    valid_label_mask = label_numeric.isin([0, 1])
    n_dropped_for_label = int((~valid_label_mask).sum())
    if n_dropped_for_label:
        print(f"[train_baseline] Dropping {n_dropped_for_label:,} of {n_rows_before:,} rows with "
              f"missing/invalid '{LABEL_COLUMN}' (e.g. a test-split file with no ground truth) "
              f"-- these cannot be used for supervised training or evaluation.")
    df = df.loc[valid_label_mask].copy()
    df[LABEL_COLUMN] = label_numeric.loc[valid_label_mask].astype(int)

    stats = {
        "input_path": str(input_path),
        "n_rows_loaded": n_rows_before,
        "n_rows_used": len(df),
        "n_rows_dropped_invalid_label": n_dropped_for_label,
        "n_feature_columns": len(feature_columns),
        "feature_columns": feature_columns,
        "n_positive": int((df[LABEL_COLUMN] == 1).sum()),
        "n_negative": int((df[LABEL_COLUMN] == 0).sum()),
        "missing_sentinel_counts": {
            col: int((df[col] == MISSING).sum()) for col in feature_columns
        },
    }
    return df, feature_columns, stats


def build_pipeline(seed: int) -> Pipeline:
    """
    Phase 3's MISSING = -1 sentinel is handled explicitly here, not treated as
    a normal similarity value: SimpleImputer(missing_values=-1, ...) is told
    exactly which value means "no data", replaces it with the column's
    median computed from the remaining (non-sentinel) training values, and
    add_indicator=True appends one extra binary column per feature that
    actually had missing values -- so the model can learn "this comparison
    was impossible" as its own signal, separately from the imputed value.
    All fitting happens inside the Pipeline, so imputation statistics and
    scaling are fit ONLY on the training split when used inside
    train_test_split + pipeline.fit (never on validation data).
    """
    return Pipeline([
        ("impute_missing_sentinel", SimpleImputer(missing_values=MISSING, strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)),
    ])


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray, grid_step: float = 0.01) -> Tuple[float, Dict]:
    """
    Selection criterion (documented, applied to the VALIDATION set only):
    maximize F1 over a grid of thresholds from 0.01 to 0.99. F1 is chosen
    because entity matching cares about precision and recall jointly (per
    spec), and F1 is their standard balanced combination -- a simple,
    interpretable criterion appropriate for a BASELINE, not a
    business-specific cost-weighted optimum. Ties are broken by picking the
    threshold closest to 0.5, the neutral default, so the choice isn't
    arbitrarily pulled to an extreme by a tie.
    """
    grid = np.arange(0.01, 1.0, grid_step)
    best = None
    for t in grid:
        result = evaluate_at_threshold(y_true, y_proba, t)
        key = (result["f1"], -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, result)
    return best[1]["threshold"], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 baseline classifier")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the Phase 3 feature CSV (default: features/train_pair_features.csv)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = Path(args.input) if args.input else project_root / "features" / "train_pair_features.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"[train_baseline] ERROR: {input_path} not found. "
                          f"Run src/build_feature_dataset.py (Phase 3) first.")

    print(f"[train_baseline] Loading {input_path} ...")
    df, feature_columns, stats = load_dataset(input_path)
    print(f"  -> {stats['n_rows_used']:,} usable rows, {stats['n_feature_columns']} feature columns, "
          f"{stats['n_positive']:,} positive / {stats['n_negative']:,} negative")

    if df[LABEL_COLUMN].nunique() < 2:
        raise SystemExit("[train_baseline] ERROR: only one class present after loading -- "
                          "cannot train/evaluate a binary classifier. Generate more Phase 3 "
                          "candidate pairs (larger --sample-size) before running Phase 4.")

    X = df[feature_columns].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[train_baseline] Train/validation split: {len(X_train)} train / {len(X_val)} validation "
          f"(stratified, seed={args.seed})")

    pipeline = build_pipeline(args.seed)

    all_missing_in_train = [
        feature_columns[i] for i in range(X_train.shape[1])
        if np.all(X_train[:, i] == MISSING)
    ]
    if all_missing_in_train:
        print(f"[train_baseline] NOTE: {len(all_missing_in_train)} feature(s) are MISSING (-1) for "
              f"every training row this run: {all_missing_in_train}. SimpleImputer has no observed "
              f"value to impute a median from, so it drops these columns from the model for this "
              f"run (a sklearn UserWarning below confirms this) -- documented in the report rather "
              f"than left as a console-only warning.")

    t0 = time.time()
    pipeline.fit(X_train, y_train)  # imputer + scaler statistics fit ONLY on X_train here
    fit_seconds = time.time() - t0

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    default_metrics = evaluate_at_threshold(y_val, y_val_proba, 0.5)
    try:
        default_metrics["roc_auc"] = round(roc_auc_score(y_val, y_val_proba), 4)
        default_metrics["pr_auc"] = round(average_precision_score(y_val, y_val_proba), 4)
    except ValueError as e:
        default_metrics["roc_auc"] = None
        default_metrics["pr_auc"] = None
        print(f"[train_baseline] WARNING: ROC-AUC/PR-AUC undefined on this validation split ({e}); "
              f"likely too few validation examples of one class. Reported as null.")

    selected_threshold, selected_metrics = select_threshold(y_val, y_val_proba)
    selected_metrics["roc_auc"] = default_metrics["roc_auc"]  # AUC metrics are threshold-independent
    selected_metrics["pr_auc"] = default_metrics["pr_auc"]

    print("=" * 70)
    print("PHASE 4 BASELINE — VALIDATION RESULTS")
    print("=" * 70)
    print(f"Default threshold (0.50): {default_metrics}")
    print(f"Selected threshold ({selected_threshold:.2f}, max-F1): {selected_metrics}")

    report = build_markdown_report(args, input_path, stats, feature_columns, fit_seconds,
                                    default_metrics, selected_threshold, selected_metrics,
                                    len(X_train), len(X_val), all_missing_in_train)
    report_path = reports_dir / "phase4_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")
    print("STOP — Phase 4 baseline only. No advanced model, no Phase 5 work.")


def build_markdown_report(args, input_path, stats, feature_columns, fit_seconds,
                           default_metrics, selected_threshold, selected_metrics,
                           n_train, n_val, all_missing_in_train) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Phase 4 Baseline Classifier\n")
    a(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    a("## Dataset\n")
    a(f"- Input file: `{input_path}`")
    a(f"- Rows loaded: {stats['n_rows_loaded']:,}")
    a(f"- Rows dropped (missing/invalid label): {stats['n_rows_dropped_invalid_label']:,}")
    a(f"- Rows used: {stats['n_rows_used']:,}")
    a(f"- Feature count: {stats['n_feature_columns']} (`{', '.join(feature_columns)}`)")
    a(f"- Identifier columns excluded from features: `{', '.join(ID_COLUMNS)}`\n")

    a("## Class Distribution\n")
    total = stats["n_positive"] + stats["n_negative"]
    pos_pct = round(stats["n_positive"] / total * 100, 2) if total else 0
    neg_pct = round(100 - pos_pct, 2) if total else 0
    a(f"- Positive count: {stats['n_positive']:,} ({pos_pct}%)")
    a(f"- Negative count: {stats['n_negative']:,} ({neg_pct}%)\n")

    a("## Model\n")
    a("- Model used: `LogisticRegression(class_weight=\"balanced\")`")
    a("- Preprocessing: `SimpleImputer(missing_values=-1, strategy=\"median\", add_indicator=True)` "
      "-> `StandardScaler()`, inside a single `sklearn.pipeline.Pipeline`")
    a("- Class imbalance handling: `class_weight=\"balanced\"` (inverse-frequency reweighting), "
      "no oversampling/undersampling introduced")
    a(f"- Random seed: {args.seed}")
    a(f"- Train/validation split: {n_train} / {n_val} rows (test_size={args.test_size}, stratified)")
    a(f"- Fit time: {fit_seconds:.3f}s\n")

    a("## Missing Sentinel Handling\n")
    a("Phase 3's `MISSING = -1` sentinel means \"this comparison could not be made\" (e.g. no "
      "postal code was parsed on one or both sides) -- it is NOT a normal similarity value and "
      "must never be treated as \"0 = confirmed non-match\" by the model. Handling used here:\n")
    a("1. `SimpleImputer(missing_values=-1, ...)` is told explicitly that `-1` (not `NaN`) marks "
      "missing values for these columns.")
    a("2. Each `-1` is replaced by that feature's **median computed only from the non-sentinel "
      "values in the training split** (fit on `X_train` only, applied unchanged to `X_val` -- no "
      "leakage).")
    a("3. `add_indicator=True` appends one extra binary column per feature that had at least one "
      "sentinel in training data, so the model can separately learn \"this comparison was "
      "impossible\" as its own signal, distinct from whatever the imputed value happens to be.")
    a("4. Phase 3's stored CSV values were not modified — the sentinel is only reinterpreted "
      "inside this Pipeline at train/predict time.\n")
    a("Missing-sentinel counts observed in the loaded data (feature: count):\n")
    for col, cnt in stats["missing_sentinel_counts"].items():
        if cnt:
            a(f"- `{col}`: {cnt:,}")
    if all_missing_in_train:
        a(f"\n**Data-quality note:** {len(all_missing_in_train)} feature(s) were `MISSING` (-1) for "
          f"every row in this run's training split — `{', '.join(all_missing_in_train)}`. "
          f"`SimpleImputer` has no observed value to compute a median from in that case, so it drops "
          f"these columns for this run (this is standard scikit-learn behavior for an all-missing "
          f"column, not a bug). This is a symptom of the small `--sample-size` used to generate the "
          f"underlying feature file, not a flaw in the feature definition itself — re-running Phase 3 "
          f"with more rows should populate these columns.")
    a("")

    a("## Validation Results\n")
    a("### At default threshold (0.50)\n")
    a(f"- Accuracy: {default_metrics['accuracy']}")
    a(f"- Precision: {default_metrics['precision']}")
    a(f"- Recall: {default_metrics['recall']}")
    a(f"- F1: {default_metrics['f1']}")
    a(f"- ROC-AUC: {default_metrics['roc_auc']}")
    a(f"- PR-AUC: {default_metrics['pr_auc']}")
    cm = default_metrics["confusion_matrix"]
    a(f"- Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}\n")

    a("## Threshold Selection\n")
    a("**Criterion:** maximize F1 over a 0.01-step grid of thresholds from 0.01 to 0.99, evaluated "
      "ONLY on the validation set (never on a held-out test set, since none was tuned against "
      "here). F1 is used because entity matching needs precision and recall balanced jointly, per "
      "spec, and F1 is the standard, interpretable way to balance them for a baseline -- not a "
      "business-specific cost model. Ties are broken by choosing the threshold closest to 0.50.\n")
    a(f"- Default (0.50) F1: {default_metrics['f1']}")
    a(f"- Selected threshold: {selected_threshold:.2f}")
    a(f"- Selected-threshold Accuracy: {selected_metrics['accuracy']}")
    a(f"- Selected-threshold Precision: {selected_metrics['precision']}")
    a(f"- Selected-threshold Recall: {selected_metrics['recall']}")
    a(f"- Selected-threshold F1: {selected_metrics['f1']}")
    cm2 = selected_metrics["confusion_matrix"]
    a(f"- Selected-threshold confusion matrix: TN={cm2['tn']}, FP={cm2['fp']}, "
      f"FN={cm2['fn']}, TP={cm2['tp']}\n")

    a("## Limitations\n")
    a("- This is a baseline model (Logistic Regression) intended to establish a floor for how well "
      "the Phase 3 engineered features separate matches from non-matches -- not a production "
      "classifier, and hyperparameters were not extensively tuned.")
    a("- Achievable recall is capped by Phase 2's blocking recall: a true match that blocking never "
      "produced as a candidate cannot appear in this dataset at all, so no classifier trained on it "
      "can recover that pair. This model's recall reflects performance only on the subset of true "
      "matches that blocking actually surfaced as candidates.")
    a("- Validation performance here is measured on a single stratified split of the current "
      "feature-generation run and does not represent final production performance, especially if "
      "this was run on a small `--sample-size` rather than the full dataset.")
    a("- Missing-sentinel rates (above) reflect the heuristic, country-agnostic postal-code/"
      "house-number extraction from Phase 2/3; features with high missingness contribute mostly "
      "through their indicator columns rather than their imputed values.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
