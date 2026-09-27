"""
Phase 5 — Stronger tree-based model

Trains a HistGradientBoostingClassifier (scikit-learn's native gradient-
boosted trees — no new dependency) on the SAME Phase 3 candidate-pair
features and the SAME stratified train/validation split methodology as
Phase 4, then reports it side-by-side with the Phase 4 Logistic Regression
baseline (refit on this run's identical split, for a fair, apples-to-apples
comparison) so the improvement (or lack of it) is directly attributable to
the model change, not to a different data split.

Nothing from Phase 3 or Phase 4 is redefined here — this script IMPORTS and
reuses:
    - features.FEATURE_NAMES, features.MISSING           (Phase 3, unchanged)
    - train_baseline.load_dataset                         (Phase 4, unchanged)
    - train_baseline.build_pipeline                        (Phase 4, unchanged — the baseline model)
    - train_baseline.evaluate_at_threshold                 (Phase 4, unchanged)
    - train_baseline.select_threshold                      (Phase 4, unchanged)
    - train_baseline.find_project_root / ID_COLUMNS / LABEL_COLUMN / RANDOM_SEED_DEFAULT

Why HistGradientBoostingClassifier:
    - Already part of scikit-learn (no xgboost/lightgbm dependency to add).
    - Histogram-based -- practical on ~1.2M rows / 16 features without a
      large memory footprint (this is explicitly why it was chosen over,
      say, a plain un-binned GradientBoostingClassifier or a large
      RandomForest with many deep trees).
    - Natively supports missing values (NaN) in split-finding, which maps
      cleanly onto Phase 3's MISSING=-1 sentinel once converted to NaN --
      no imputation/indicator-column preprocessing is needed for this model
      (unlike the Phase 4 Logistic Regression baseline, which does need
      that, since linear models can't take NaN as an input).
    - class_weight="balanced" is supported directly, handling the severe
      class imbalance (297 positive / 1,198,831 negative on the real
      dataset -- roughly 1:4037) without oversampling/undersampling.

Usage:
    python src/train_model_v2.py
    python src/train_model_v2.py --input features/train_pair_features.csv --test-size 0.2 --seed 42

Writes:
    reports/phase5_model_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import MISSING
from train_baseline import (
    ID_COLUMNS,
    LABEL_COLUMN,
    RANDOM_SEED_DEFAULT,
    build_pipeline as build_baseline_pipeline,
    evaluate_at_threshold,
    find_project_root,
    load_dataset,
    select_threshold,
)

# Phase 7 fix: when this file is run directly (`python train_model_v2.py`),
# Python executes it as module "__main__", not "train_model_v2" -- so a
# joblib-pickled object referencing one of this module's functions (e.g.
# FunctionTransformer(_sentinel_to_nan) inside build_gbm_pipeline) would be
# unloadable from any OTHER script, which imports this file under its real
# name. Registering this exact, currently-running module object under BOTH
# names in sys.modules means pickle's identity check
# (getattr(sys.modules['train_model_v2'], name) is obj) succeeds regardless
# of which name the script happened to run under. This changes no runtime
# behavior of the module -- it only fixes cross-script model persistence.
sys.modules.setdefault("train_model_v2", sys.modules[__name__])

DEFAULT_MAX_ITER = 200  # fixed, sensible default -- NOT tuned/swept (per spec)


def _sentinel_to_nan(X: np.ndarray) -> np.ndarray:
    """
    Converts Phase 3's MISSING=-1 sentinel to np.nan, IN A COPY (never
    mutates the caller's array). HistGradientBoostingClassifier natively
    treats NaN as "missing" during split-finding, learning which branch
    missing values should go down at each split -- a tree-native analogue
    of Phase 4's SimpleImputer(add_indicator=True), suited to a model that
    can use missingness directly as signal without needing an imputed
    numeric stand-in.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X.copy()
    X[X == MISSING] = np.nan
    return X


# Phase 7 fix: when this script is executed directly (`python train_model_v2.py`),
# Python sets this module's __name__/__module__ to "__main__", so a joblib-
# pickled FunctionTransformer(_sentinel_to_nan) would record its callable's
# module as "__main__" -- which breaks unpickling from any OTHER script
# (e.g. src/inference.py), since Python can't find "_sentinel_to_nan" on a
# fresh "__main__" module there. Explicitly pinning __module__ to this
# file's real importable name fixes cross-script loading without changing
# _sentinel_to_nan's behavior at all.
_sentinel_to_nan.__module__ = "train_model_v2"


def build_gbm_pipeline(seed: int, max_iter: int = DEFAULT_MAX_ITER) -> Pipeline:
    """
    Phase 5 model pipeline: sentinel->NaN conversion, then
    HistGradientBoostingClassifier. No scaling step -- tree splits are
    invariant to monotonic feature scaling, so StandardScaler (used in the
    Phase 4 linear-model pipeline) would be redundant here.

    early_stopping is explicitly disabled: HistGradientBoostingClassifier's
    default early_stopping="auto" would carve out its OWN internal
    validation slice from X_train, which is a second, hidden train/val
    split Phase 4 doesn't have. Disabling it keeps this pipeline's only
    train/validation split identical to Phase 4's external stratified
    split, so the two models' validation numbers are directly comparable.
    max_iter is a small fixed value (not swept) to keep training bounded on
    ~1.2M rows without an expensive search.
    """
    return Pipeline([
        ("sentinel_to_nan", FunctionTransformer(_sentinel_to_nan, validate=False)),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=max_iter,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )),
    ])


def _auc_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (None, None) when ROC-AUC/PR-AUC are undefined (only one class
    present in y_true) rather than a spurious number. Checked explicitly
    up front rather than relying on an exception: in this scikit-learn
    version, roc_auc_score/average_precision_score emit an
    UndefinedMetricWarning and return NaN/0.0 in that case instead of
    raising -- silently keeping either as the "score" would be misleading.
    """
    if len(np.unique(y_true)) < 2:
        return None, None
    return round(roc_auc_score(y_true, y_proba), 4), round(average_precision_score(y_true, y_proba), 4)


def _evaluate_model(name: str, pipeline: Pipeline, X_train, y_train, X_val, y_val) -> Dict:
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    default_metrics = evaluate_at_threshold(y_val, y_val_proba, 0.5)
    roc_auc, pr_auc = _auc_metrics(y_val, y_val_proba)
    default_metrics["roc_auc"] = roc_auc
    default_metrics["pr_auc"] = pr_auc

    selected_threshold, selected_metrics = select_threshold(y_val, y_val_proba)
    selected_metrics["roc_auc"] = roc_auc
    selected_metrics["pr_auc"] = pr_auc

    return {
        "name": name,
        "fit_seconds": round(fit_seconds, 3),
        "default_metrics": default_metrics,
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "fitted_pipeline": pipeline,  # Phase 7 addition: needed to persist the trained model for inference
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 tree-based model")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the Phase 3 feature CSV (default: features/train_pair_features.csv)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                         help="Fixed boosting-round count for the Phase 5 model (not tuned/swept).")
    parser.add_argument("--no-save-model", action="store_true",
                         help="Phase 7 addition: skip persisting the fitted Phase 5 pipeline to models/. "
                              "By default the model IS saved, since Phase 7's inference module loads it.")
    parser.add_argument("--model-dir", type=str, default=None,
                         help="Phase 7 addition: directory to save the trained model into (default: models/).")
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = Path(args.input) if args.input else project_root / "features" / "train_pair_features.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"[train_model_v2] ERROR: {input_path} not found. "
                          f"Run src/build_feature_dataset.py (Phase 3) first.")

    print(f"[train_model_v2] Loading {input_path} ...")
    df, feature_columns, stats = load_dataset(input_path)  # Phase 4's loader, reused unchanged
    print(f"  -> {stats['n_rows_used']:,} usable rows, {stats['n_feature_columns']} feature columns, "
          f"{stats['n_positive']:,} positive / {stats['n_negative']:,} negative")

    if df[LABEL_COLUMN].nunique() < 2:
        raise SystemExit("[train_model_v2] ERROR: only one class present after loading -- "
                          "cannot train/evaluate a binary classifier.")

    # Load once, as float32 to halve the in-memory footprint versus the
    # pandas float64 default -- there is exactly one X array for this whole
    # run; both models below are fit on the SAME X_train/y_train, no
    # per-model copy of the full dataset is made.
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy()
    del df  # the DataFrame is no longer needed once X/y are extracted

    # Identical split call to Phase 4 (train_baseline.py): same test_size
    # default, same stratification, same seed default -- preserves Phase 4's
    # validation methodology exactly, per spec.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[train_model_v2] Train/validation split: {len(X_train)} train / {len(X_val)} validation "
          f"(stratified, seed={args.seed}) -- identical split methodology to Phase 4")

    print("[train_model_v2] Fitting Phase 4 baseline on this run's split (for a same-split comparison) ...")
    baseline_result = _evaluate_model(
        "Phase 4 baseline (LogisticRegression)",
        build_baseline_pipeline(args.seed), X_train, y_train, X_val, y_val,
    )

    print(f"[train_model_v2] Fitting Phase 5 model (HistGradientBoostingClassifier, max_iter={args.max_iter}) ...")
    phase5_result = _evaluate_model(
        "Phase 5 model (HistGradientBoostingClassifier)",
        build_gbm_pipeline(args.seed, args.max_iter), X_train, y_train, X_val, y_val,
    )

    print("=" * 70)
    print("PHASE 5 — VALIDATION RESULTS")
    print("=" * 70)
    for result in (baseline_result, phase5_result):
        print(f"{result['name']}:")
        print(f"  default(0.50): {result['default_metrics']}")
        print(f"  selected({result['selected_threshold']:.2f}): {result['selected_metrics']}")

    report = build_markdown_report(args, input_path, stats, feature_columns,
                                    len(X_train), len(X_val), baseline_result, phase5_result)
    report_path = reports_dir / "phase5_model_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")

    if not args.no_save_model:
        # Phase 7 addition: persist the already-fitted Phase 5 pipeline so
        # src/inference.py can load it without retraining. This does not
        # change any Phase 5 metric, methodology, or report content above --
        # it saves the exact pipeline object that was already fit and
        # evaluated. See docs/PHASE7_DESIGN.md for the full contract.
        import joblib

        model_dir = Path(args.model_dir) if args.model_dir else project_root / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "phase5_model.joblib"
        metadata_path = model_dir / "phase5_model_metadata.json"

        joblib.dump(phase5_result["fitted_pipeline"], model_path)
        metadata = {
            "model_type": "HistGradientBoostingClassifier (Phase 5)",
            "feature_columns": feature_columns,
            "default_threshold": 0.5,
            "phase5_selected_threshold": phase5_result["selected_threshold"],
            "phase5_selected_threshold_criterion": "max-F1 on validation set",
            "training_input_file": str(input_path),
            "training_rows_used": stats["n_rows_used"],
            "seed": args.seed,
            "test_size": args.test_size,
            "max_iter": args.max_iter,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Model saved: {model_path}")
        print(f"Model metadata saved: {metadata_path}")

    print("STOP — Phase 5 only. No deployment/API/ranking, no Phase 6 work.")


def build_markdown_report(args, input_path, stats, feature_columns, n_train, n_val,
                           baseline_result: Dict, phase5_result: Dict) -> str:
    lines: List[str] = []
    a = lines.append

    def fmt(v):
        return "N/A" if v is None else v

    a("# Phase 5 Model Report\n")
    a(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    a("## Dataset\n")
    a(f"- Input file: `{input_path}`")
    a(f"- Rows used: {stats['n_rows_used']:,}")
    a(f"- Feature count: {stats['n_feature_columns']} (`{', '.join(feature_columns)}`) "
      f"-- unchanged from Phase 3/4\n")

    a("## Class Distribution\n")
    total = stats["n_positive"] + stats["n_negative"]
    pos_pct = round(stats["n_positive"] / total * 100, 4) if total else 0
    a(f"- Positive count: {stats['n_positive']:,} ({pos_pct}%)")
    a(f"- Negative count: {stats['n_negative']:,} ({round(100 - pos_pct, 4)}%)")
    a(f"- Train/validation split: {n_train:,} / {n_val:,} rows "
      f"(test_size={args.test_size}, stratified, seed={args.seed} -- identical methodology to Phase 4)\n")

    a("## Models Compared\n")
    a("### Phase 4 baseline (refit on this run's identical split)\n")
    a("- `LogisticRegression(class_weight=\"balanced\")` inside a "
      "`SimpleImputer(missing_values=-1, add_indicator=True) -> StandardScaler` Pipeline "
      "(`train_baseline.build_pipeline`, imported unchanged).\n")
    a("### Phase 5 model\n")
    a("- `HistGradientBoostingClassifier(class_weight=\"balanced\", early_stopping=False, "
      f"max_iter={args.max_iter})` -- scikit-learn's native histogram-based gradient boosting; "
      "no new dependency (xgboost/lightgbm were deliberately not introduced).")
    a("- Chosen over a plain `GradientBoostingClassifier` or a large `RandomForestClassifier` "
      "specifically for memory/practicality on ~1.2M rows: histogram binning keeps memory bounded "
      "regardless of row count, unlike a forest of many deep, unbinned trees.")
    a(f"- Random seed: {args.seed} (same seed as the baseline run above)\n")

    a("## Missing Sentinel Handling\n")
    a("Phase 3's `MISSING = -1` sentinel means \"this comparison could not be made\" and must never "
      "be treated as a normal similarity value (unchanged principle from Phase 4). Handling differs "
      "by model family, not by redefinition of the sentinel itself:\n")
    a("- **Baseline (linear model):** `-1` is imputed to the training-split median plus a "
      "missing-indicator column (Phase 4's approach, reused unchanged here).")
    a("- **Phase 5 (tree model):** `-1` is converted to `np.nan` (a pure representation change, not "
      "a value change), and `HistGradientBoostingClassifier` natively learns, per split, which branch "
      "missing values should follow — using missingness as signal directly rather than through a "
      "separate indicator column.")
    a("- Neither approach modifies Phase 3's stored CSV values; both reinterpret the sentinel only "
      "inside their respective Pipelines at train/predict time.\n")

    a("## Validation Results\n")
    for result in (baseline_result, phase5_result):
        dm, sm = result["default_metrics"], result["selected_metrics"]
        a(f"### {result['name']}\n")
        a(f"Fit time: {result['fit_seconds']}s\n")
        a("**At default threshold (0.50):**")
        a(f"- Accuracy: {fmt(dm['accuracy'])}  Precision: {fmt(dm['precision'])}  "
          f"Recall: {fmt(dm['recall'])}  F1: {fmt(dm['f1'])}")
        a(f"- ROC-AUC: {fmt(dm['roc_auc'])}  PR-AUC: {fmt(dm['pr_auc'])}")
        cm = dm["confusion_matrix"]
        a(f"- Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}\n")
        a(f"**At selected threshold ({result['selected_threshold']:.2f}, max-F1 on validation set):**")
        a(f"- Accuracy: {fmt(sm['accuracy'])}  Precision: {fmt(sm['precision'])}  "
          f"Recall: {fmt(sm['recall'])}  F1: {fmt(sm['f1'])}")
        cm2 = sm["confusion_matrix"]
        a(f"- Confusion matrix: TN={cm2['tn']}, FP={cm2['fp']}, FN={cm2['fn']}, TP={cm2['tp']}\n")

    a("## Phase 5 vs Phase 4 Comparison (same split, selected threshold for each model)\n")
    b_sm, p_sm = baseline_result["selected_metrics"], phase5_result["selected_metrics"]

    def delta(k, higher_is_better=True):
        bv, pv = b_sm.get(k), p_sm.get(k)
        if bv is None or pv is None:
            return "N/A"
        d = round(pv - bv, 4)
        sign = "+" if d >= 0 else ""
        verdict = "better" if (d > 0) == higher_is_better and d != 0 else ("worse" if d != 0 else "unchanged")
        return f"{sign}{d} ({verdict})"

    a(f"- Precision: baseline={fmt(b_sm['precision'])} -> phase5={fmt(p_sm['precision'])} ({delta('precision')})")
    a(f"- Recall: baseline={fmt(b_sm['recall'])} -> phase5={fmt(p_sm['recall'])} ({delta('recall')})")
    a(f"- F1: baseline={fmt(b_sm['f1'])} -> phase5={fmt(p_sm['f1'])} ({delta('f1')})")
    a(f"- ROC-AUC: baseline={fmt(b_sm['roc_auc'])} -> phase5={fmt(p_sm['roc_auc'])} ({delta('roc_auc')})")
    a(f"- PR-AUC: baseline={fmt(b_sm['pr_auc'])} -> phase5={fmt(p_sm['pr_auc'])} ({delta('pr_auc')})")
    if pos_pct < 5:
        a("\nPR-AUC is the most informative single number here given the severity of the class "
          "imbalance (positive rate well under 5%) — ROC-AUC can look deceptively high under heavy "
          "imbalance because true negatives dominate.\n")
    else:
        a("")

    a("## Limitations\n")
    a("- Neither model can recover a true match that Phase 2's blocking never produced as a "
      "candidate in the first place — the achievable recall ceiling from Phase 2/3 is unchanged "
      "and applies identically to both models here.")
    a("- `max_iter` for the Phase 5 model was fixed, not tuned via search, per the Phase 5 "
      "instruction to avoid expensive hyperparameter sweeps; further gains are plausible from " 
      "modest tuning (e.g. `max_leaf_nodes`, `learning_rate`) but that is explicitly out of scope here.")
    a("- With positives this rare, validation metrics (especially precision) can be volatile — a "
      "handful of additional false positives/negatives can move precision/F1 substantially. Treat "
      "point estimates here as noisy, not exact.")
    a("- This comparison used a single stratified split; it does not by itself establish that "
      "Phase 5 generalizes better across other splits/seeds (that would need repeated/k-fold "
      "evaluation, out of scope for this baseline-vs-improved-model comparison).")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
