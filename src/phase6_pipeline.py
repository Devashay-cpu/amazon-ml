"""
Phase 6 — Threshold / business-cost calibration + second-stage recall
recovery investigation

Two independent analyses, one combined report:

  PART A — Threshold / business-cost calibration
    Refits the Phase 4 baseline and Phase 5 model on the IDENTICAL split
    methodology already used in Phase 4/5 (same seed, same test_size,
    same stratification), then sweeps a threshold grid over Phase 5's
    validation probabilities, scoring each threshold with an explicit,
    documented business-cost function (cost_fp * FP + cost_fn * FN)
    instead of silently assuming 0.50 or re-deriving Phase 5's own
    max-F1 threshold. Optionally (--tune, and only if Phase 5 already
    materially beats Phase 4) runs a SMALL RandomizedSearchCV over
    max_leaf_nodes / learning_rate / min_samples_leaf, fit ONLY on the
    training split (X_val is never touched by the search).

  PART B — Blocking recall recovery investigation
    Reads Phase 2's reports/blocking_stats.json (produced by
    evaluate_blocking.py) and summarizes which blocking strategies
    contribute the most/least to recall and which blocks are
    oversized. Optionally (--relax-experiment), runs ONE targeted,
    small-scale relaxation of the rare-name-token blocking strategy
    (by temporarily monkeypatching blocking.py's module-level
    RARE_TOKEN_MAX_DOC_FRACTION / RARE_TOKEN_MAX_DOC_CAP constants --
    blocking.py itself is NEVER modified) and reports the recall and
    candidate-volume delta versus the baseline run, so a relaxation is
    only actually tried when explicitly requested, on top of the
    already-computed baseline evidence -- never blindly.

Both parts reuse Phase 1-5 code unchanged:
    train_baseline.load_dataset / evaluate_at_threshold / select_threshold
    train_baseline.build_pipeline               (Phase 4 baseline model)
    train_model_v2.build_gbm_pipeline            (Phase 5 model)
    candidate_generation.build_index_from_source / iter_candidates
    gt_loader.load_ground_truth
    evaluate_blocking.candidate_count_stats / percentile / safe_div

Usage:
    python src/phase6_pipeline.py
    python src/phase6_pipeline.py --tune                    # optional light tuning, gated (see above)
    python src/phase6_pipeline.py --relax-experiment        # optional blocking relaxation experiment
    python src/phase6_pipeline.py --cost-fp 1 --cost-fn 20  # custom business-cost weights

Writes:
    reports/phase6_calibration_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from evaluate_blocking import candidate_count_stats, safe_div
from features import FEATURE_NAMES
from train_baseline import (
    LABEL_COLUMN,
    RANDOM_SEED_DEFAULT,
    build_pipeline as build_baseline_pipeline,
    evaluate_at_threshold,
    find_project_root,
    load_dataset,
    select_threshold,
)
from train_model_v2 import _auc_metrics, build_gbm_pipeline

# --------------------------------------------------------------------------
# PART A — business cost + threshold calibration
# --------------------------------------------------------------------------

# DOCUMENTED, CONFIGURABLE ASSUMPTION (override via --cost-fp / --cost-fn):
# a missed true match (false negative) is treated as 10x costlier than an
# incorrect merge (false positive). Rationale: in this two-stage
# entity-resolution pipeline, a false positive is a candidate pair a human
# reviewer or a downstream QA process can still catch and reverse, whereas
# a false negative silently leaves two records unmerged -- the error is
# invisible unless someone happens to go looking for it, and duplicate
# entities then quietly corrupt any downstream count/aggregation that
# assumes one row = one business. This ratio is NOT a measured business
# figure; it is a starting assumption to be replaced with the project's
# real cost figures once known.
DEFAULT_COST_FP = 1.0
DEFAULT_COST_FN = 10.0

TUNE_GATE_MARGIN = 0.05  # only tune if Phase5 PR-AUC beats Phase4 PR-AUC by at least this much
TUNE_N_ITER = 8          # small, fixed search budget -- not an expensive sweep
TUNE_CV = 3


def business_cost(n_fp: int, n_fn: int, cost_fp: float, cost_fn: float) -> float:
    """Total business cost for a confusion matrix: cost_fp * FP + cost_fn * FN.
    Pure function of counts and weights -- deterministic, no hidden state."""
    return cost_fp * n_fp + cost_fn * n_fn


def analyze_thresholds(
    y_val: np.ndarray, y_proba: np.ndarray, cost_fp: float, cost_fn: float, grid_step: float = 0.01
) -> Tuple[List[Dict], Dict]:
    """
    Evaluates a 0.01-step threshold grid (0.01..0.99) on the VALIDATION set
    only, attaching a business_cost to each row via evaluate_at_threshold's
    (Phase 4, reused) confusion-matrix output. Returns (all rows, the
    lowest-cost row). Ties are broken by preferring the threshold closest
    to 0.50, for the same "don't let a tie arbitrarily pick an extreme"
    reason Phase 4's own select_threshold uses.
    """
    grid = np.round(np.arange(0.01, 1.0, grid_step), 4)
    rows: List[Dict] = []
    best = None
    for t in grid:
        m = evaluate_at_threshold(y_val, y_proba, float(t))
        cm = m["confusion_matrix"]
        m["business_cost"] = business_cost(cm["fp"], cm["fn"], cost_fp, cost_fn)
        rows.append(m)
        key = (-m["business_cost"], -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, m)
    return rows, best[1]


def maybe_tune(X_train: np.ndarray, y_train: np.ndarray, seed: int,
                n_iter: int = TUNE_N_ITER, cv: int = TUNE_CV):
    """
    Small, fixed-budget RandomizedSearchCV over the Phase 5 model's own
    pipeline (build_gbm_pipeline, imported unchanged) -- searched
    hyperparameters are exactly the three named in the Phase 6 spec.
    Scoring is PR-AUC ("average_precision"), matching what actually
    matters for this imbalanced task. Cross-validation folds are drawn
    ONLY from X_train/y_train -- X_val is never passed to .fit() or seen
    by the search in any way, so no validation information can leak into
    the tuned hyperparameters.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier  # local import: only needed if tuning runs

    param_dist = {
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__min_samples_leaf": [10, 20, 50],
    }
    n_pos_train = int((y_train == 1).sum())
    safe_cv = max(2, min(cv, n_pos_train))  # avoid a fold with zero positives on tiny/toy data
    search = RandomizedSearchCV(
        build_gbm_pipeline(seed),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, round(float(search.best_score_), 4)


# --------------------------------------------------------------------------
# PART B — blocking recall recovery investigation
# --------------------------------------------------------------------------

def load_blocking_stats(reports_dir: Path) -> Optional[Dict]:
    path = reports_dir / "blocking_stats.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_blocking_stats(stats: Dict) -> Dict[str, Any]:
    """
    Pure function (no I/O) over an already-loaded blocking_stats.json
    (Phase 2's evaluate_blocking.py output): ranks strategies by true-pair
    contribution, flags oversized blocks, and reports recall by group.
    Kept separate from load_blocking_stats so it's independently unit
    -testable against a small synthetic dict.
    """
    recall = stats.get("blocking_recall") or {}
    strategy_pairs = stats.get("strategy_pairs_found") or {}
    ranked_strategies = sorted(strategy_pairs.items(), key=lambda kv: -kv[1])

    s2_problem = stats.get("s2_problematic_blocks") or []
    s3_problem = stats.get("s3_problematic_blocks") or []

    low_value_strategies = [
        s for s, n in ranked_strategies
        if n == 0 or n < 0.01 * max(strategy_pairs.values(), default=1)
    ]

    return {
        "recall": recall,
        "ranked_strategies": ranked_strategies,
        "low_value_strategies": low_value_strategies,
        "n_s2_problematic_blocks": len(s2_problem),
        "n_s3_problematic_blocks": len(s3_problem),
        "top_s2_problematic_blocks": s2_problem[:5],
        "top_s3_problematic_blocks": s3_problem[:5],
        "run_entities_evaluated": (stats.get("run") or {}).get("entities_evaluated"),
    }


def run_relaxed_blocking_experiment(project_root: Path, dataset: str, sample_size: Optional[int],
                                     relaxed_fraction: float, relaxed_cap: int) -> Dict[str, Any]:
    """
    Runs baseline vs. relaxed blocking side by side and reports the recall
    and candidate-volume delta. blocking.py is NEVER modified -- its
    module-level RARE_TOKEN_MAX_DOC_FRACTION / RARE_TOKEN_MAX_DOC_CAP
    constants (read at call time inside BlockIndex.finalize_pass1) are
    temporarily monkeypatched for the "relaxed" pass only, then restored,
    so Phase 2's own default behavior is completely unaffected by running
    this experiment.
    """
    import blocking as blocking_module
    from candidate_generation import build_index_from_source, iter_candidates
    from gt_loader import load_ground_truth

    dataset_dir = project_root / "dataset" / dataset
    s1_path = dataset_dir / f"{dataset}_source1.tsv"
    s2_path = dataset_dir / f"{dataset}_source2.tsv"
    s3_path = dataset_dir / f"{dataset}_source3.tsv"
    gt_path = project_root / "dataset" / "train" / "train_ground_truth.tsv"

    def _one_pass(label: str) -> Dict[str, Any]:
        s2_index = build_index_from_source(s2_path, "S2")
        s3_index = build_index_from_source(s3_path, "S3")
        ground_truth = load_ground_truth(gt_path, s2_index.all_ids, s3_index.all_ids)

        candidate_counts: List[int] = []
        gt_pairs = 0
        found_pairs = 0
        for entity_id, s2_cands, s3_cands, _, _ in iter_candidates(s1_path, s2_index, s3_index, limit=sample_size):
            candidate_counts.append(len(s2_cands) + len(s3_cands))
            gt = ground_truth.get(entity_id, {"s2": set(), "s3": set()})
            gt_pairs += len(gt["s2"]) + len(gt["s3"])
            found_pairs += len(gt["s2"] & s2_cands) + len(gt["s3"] & s3_cands)

        return {
            "label": label,
            "recall": safe_div(found_pairs, gt_pairs),
            "gt_pairs": gt_pairs,
            "found_pairs": found_pairs,
            "candidate_count_stats": candidate_count_stats(candidate_counts),
        }

    baseline_result = _one_pass("baseline (current Phase 2 defaults)")

    original_fraction = blocking_module.RARE_TOKEN_MAX_DOC_FRACTION
    original_cap = blocking_module.RARE_TOKEN_MAX_DOC_CAP
    try:
        blocking_module.RARE_TOKEN_MAX_DOC_FRACTION = relaxed_fraction
        blocking_module.RARE_TOKEN_MAX_DOC_CAP = relaxed_cap
        relaxed_result = _one_pass(
            f"relaxed (rare-token fraction={relaxed_fraction}, cap={relaxed_cap})"
        )
    finally:
        blocking_module.RARE_TOKEN_MAX_DOC_FRACTION = original_fraction
        blocking_module.RARE_TOKEN_MAX_DOC_CAP = original_cap

    return {"baseline": baseline_result, "relaxed": relaxed_result}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 threshold calibration + recall recovery investigation")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the Phase 3 feature CSV (default: features/train_pair_features.csv)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP)
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN)
    parser.add_argument("--tune", action="store_true",
                        help="Attempt small RandomizedSearchCV tuning, gated on Phase5 materially beating Phase4")
    parser.add_argument("--relax-experiment", action="store_true",
                        help="Run one targeted rare-token blocking relaxation experiment (requires dataset/ files)")
    parser.add_argument("--relaxed-fraction", type=float, default=0.002,
                        help="Relaxed RARE_TOKEN_MAX_DOC_FRACTION to try (default is 4x Phase 2's 0.0005)")
    parser.add_argument("--relaxed-cap", type=int, default=200,
                        help="Relaxed RARE_TOKEN_MAX_DOC_CAP to try (default is 4x Phase 2's 50)")
    parser.add_argument("--blocking-dataset", choices=["train", "test"], default="train")
    parser.add_argument("--blocking-sample-size", type=int, default=20_000)
    parser.add_argument("--no-save-threshold", action="store_true",
                         help="Phase 7 addition: skip persisting the selected threshold to models/. "
                              "By default it IS saved, since Phase 7's inference module loads it.")
    parser.add_argument("--model-dir", type=str, default=None,
                         help="Phase 7 addition: directory to save the selected threshold into (default: models/).")
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = Path(args.input) if args.input else project_root / "features" / "train_pair_features.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"[phase6_pipeline] ERROR: {input_path} not found. Run Phase 3/4/5 first.")

    # ---------------- PART A: threshold / business-cost calibration ----------------
    print(f"[phase6_pipeline] Loading {input_path} ...")
    df, feature_columns, stats = load_dataset(input_path)
    print(f"  -> {stats['n_rows_used']:,} usable rows, {stats['n_feature_columns']} feature columns, "
          f"{stats['n_positive']:,} positive / {stats['n_negative']:,} negative")

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy()
    del df

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[phase6_pipeline] Train/validation split: {len(X_train)} / {len(X_val)} "
          f"(identical methodology to Phase 4/5, seed={args.seed})")

    print("[phase6_pipeline] Fitting Phase 4 baseline and Phase 5 model on this split ...")
    baseline_pipeline = build_baseline_pipeline(args.seed)
    baseline_pipeline.fit(X_train, y_train)
    baseline_proba = baseline_pipeline.predict_proba(X_val)[:, 1]
    baseline_roc_auc, baseline_pr_auc = _auc_metrics(y_val, baseline_proba)

    phase5_pipeline = build_gbm_pipeline(args.seed)
    phase5_pipeline.fit(X_train, y_train)
    phase5_proba = phase5_pipeline.predict_proba(X_val)[:, 1]
    phase5_roc_auc, phase5_pr_auc = _auc_metrics(y_val, phase5_proba)

    phase5_default = evaluate_at_threshold(y_val, phase5_proba, 0.5)
    phase5_default["roc_auc"], phase5_default["pr_auc"] = phase5_roc_auc, phase5_pr_auc
    phase5_f1_threshold, phase5_f1_metrics = select_threshold(y_val, phase5_proba)
    phase5_f1_metrics["roc_auc"], phase5_f1_metrics["pr_auc"] = phase5_roc_auc, phase5_pr_auc

    print(f"[phase6_pipeline] Sweeping thresholds with business cost "
          f"(cost_fp={args.cost_fp}, cost_fn={args.cost_fn}) ...")
    threshold_rows, best_cost_row = analyze_thresholds(y_val, phase5_proba, args.cost_fp, args.cost_fn)
    best_cost_row["roc_auc"], best_cost_row["pr_auc"] = phase5_roc_auc, phase5_pr_auc

    tuning_result = None
    pr_auc_gap = (phase5_pr_auc - baseline_pr_auc) if (phase5_pr_auc is not None and baseline_pr_auc is not None) else None
    if args.tune:
        if pr_auc_gap is not None and pr_auc_gap >= TUNE_GATE_MARGIN:
            print(f"[phase6_pipeline] Phase5 PR-AUC beats Phase4 by {pr_auc_gap:.4f} "
                  f"(>= gate {TUNE_GATE_MARGIN}) -- running small tuning search ...")
            t0 = time.time()
            tuned_estimator, best_params, best_cv_pr_auc = maybe_tune(X_train, y_train, args.seed)
            tuned_proba = tuned_estimator.predict_proba(X_val)[:, 1]
            tuned_roc_auc, tuned_pr_auc = _auc_metrics(y_val, tuned_proba)
            tuned_rows, tuned_best_cost_row = analyze_thresholds(y_val, tuned_proba, args.cost_fp, args.cost_fn)
            tuned_best_cost_row["roc_auc"], tuned_best_cost_row["pr_auc"] = tuned_roc_auc, tuned_pr_auc
            tuning_result = {
                "ran": True,
                "seconds": round(time.time() - t0, 2),
                "best_params": best_params,
                "best_cv_pr_auc": best_cv_pr_auc,
                "val_roc_auc": tuned_roc_auc,
                "val_pr_auc": tuned_pr_auc,
                "best_cost_row": tuned_best_cost_row,
            }
        else:
            print(f"[phase6_pipeline] --tune requested but gate not met "
                  f"(PR-AUC gap={pr_auc_gap}, need >= {TUNE_GATE_MARGIN}) -- skipping tuning.")
            tuning_result = {"ran": False, "reason": f"PR-AUC gap {pr_auc_gap} below gate {TUNE_GATE_MARGIN}"}

    # ---------------- PART B: blocking recall recovery investigation ----------------
    print("[phase6_pipeline] Loading Phase 2 blocking_stats.json for recall investigation ...")
    raw_blocking_stats = load_blocking_stats(reports_dir)
    blocking_summary = summarize_blocking_stats(raw_blocking_stats) if raw_blocking_stats else None
    if blocking_summary is None:
        print("  -> reports/blocking_stats.json not found -- run src/evaluate_blocking.py first "
              "for a real blocking-recall analysis. Proceeding with this section empty.")

    relax_result = None
    if args.relax_experiment:
        dataset_dir = project_root / "dataset" / args.blocking_dataset
        if not (dataset_dir / f"{args.blocking_dataset}_source1.tsv").exists():
            print("[phase6_pipeline] --relax-experiment requested but dataset files not found -- skipping.")
        else:
            print(f"[phase6_pipeline] Running targeted rare-token relaxation experiment "
                  f"(fraction={args.relaxed_fraction}, cap={args.relaxed_cap}) ...")
            relax_result = run_relaxed_blocking_experiment(
                project_root, args.blocking_dataset, args.blocking_sample_size,
                args.relaxed_fraction, args.relaxed_cap,
            )

    report = build_markdown_report(
        args, input_path, stats, feature_columns, len(X_train), len(X_val),
        baseline_pr_auc, baseline_roc_auc,
        phase5_default, phase5_f1_threshold, phase5_f1_metrics,
        threshold_rows, best_cost_row, tuning_result,
        blocking_summary, relax_result,
    )
    report_path = reports_dir / "phase6_calibration_report.md"
    report_path.write_text(report, encoding="utf-8")

    if not args.no_save_threshold:
        # Phase 7 addition: persist the selected threshold (+ the cost
        # weights that produced it) so src/inference.py can apply it without
        # re-running this whole calibration. Purely additive -- no existing
        # computed value, report content, or file above is changed by this.
        model_dir = Path(args.model_dir) if args.model_dir else project_root / "models"
        model_dir.mkdir(exist_ok=True)
        threshold_path = model_dir / "phase6_threshold.json"
        threshold_meta = {
            "selected_threshold": best_cost_row["threshold"],
            "selection_criterion": "minimum business cost on validation set",
            "cost_fp": args.cost_fp,
            "cost_fn": args.cost_fn,
            "validation_metrics_at_threshold": {
                "precision": best_cost_row["precision"],
                "recall": best_cost_row["recall"],
                "f1": best_cost_row["f1"],
                "business_cost": best_cost_row["business_cost"],
            },
            "seed": args.seed,
            "test_size": args.test_size,
            "input_file": str(input_path),
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_meta, f, indent=2)
        print(f"Threshold saved: {threshold_path}")

    print("=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)
    print(f"Phase4 PR-AUC={baseline_pr_auc}  Phase5 PR-AUC={phase5_pr_auc}")
    print(f"Phase5 max-F1 threshold={phase5_f1_threshold:.2f}  metrics={phase5_f1_metrics}")
    print(f"Phase6 cost-selected threshold={best_cost_row['threshold']}  metrics={best_cost_row}")
    if tuning_result:
        print(f"Tuning: {tuning_result}")
    if blocking_summary:
        print(f"Blocking recall (from existing stats): {blocking_summary['recall']}")
    if relax_result:
        print(f"Relaxation experiment: {relax_result}")
    print(f"Report: {report_path}")
    print("STOP — Phase 6 only. No deployment/API/ranking, no Phase 7 work.")


def build_markdown_report(args, input_path, stats, feature_columns, n_train, n_val,
                           baseline_pr_auc, baseline_roc_auc,
                           phase5_default, phase5_f1_threshold, phase5_f1_metrics,
                           threshold_rows, best_cost_row, tuning_result,
                           blocking_summary, relax_result) -> str:
    lines: List[str] = []
    a = lines.append

    def fmt(v):
        return "N/A" if v is None else v

    a("# Phase 6 Calibration Report\n")
    a(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    a("## Phase 5 baseline metrics (this run's split)\n")
    a(f"- Input: `{input_path}`, {stats['n_rows_used']:,} rows, {stats['n_feature_columns']} features")
    a(f"- Train/val split: {n_train:,} / {n_val:,} (seed={args.seed}, test_size={args.test_size})")
    a(f"- Phase 4 baseline: ROC-AUC={fmt(baseline_roc_auc)}, PR-AUC={fmt(baseline_pr_auc)}")
    a(f"- Phase 5 model @ default 0.50: {phase5_default}")
    a(f"- Phase 5 model @ its own max-F1 threshold ({phase5_f1_threshold:.2f}): {phase5_f1_metrics}\n")

    a("## Threshold analysis table (excerpt — every 10th of 99 grid points)\n")
    a("| threshold | precision | recall | f1 | fp | fn | business_cost |")
    a("|---|---|---|---|---|---|---|")
    for row in threshold_rows[::10]:
        cm = row["confusion_matrix"]
        a(f"| {row['threshold']:.2f} | {row['precision']} | {row['recall']} | {row['f1']} | "
          f"{cm['fp']} | {cm['fn']} | {row['business_cost']:.1f} |")
    a("")

    a("## Business cost definition\n")
    a(f"`cost = {args.cost_fp} * false_positives + {args.cost_fn} * false_negatives`\n")
    a(f"- cost_fp = {args.cost_fp}, cost_fn = {args.cost_fn} "
      f"({'defaults' if (args.cost_fp == DEFAULT_COST_FP and args.cost_fn == DEFAULT_COST_FN) else 'custom, user-supplied'})")
    a("- Rationale for the default 1:10 ratio: a false positive (incorrect merge) is a candidate pair "
      "a downstream review process can still catch and reverse; a false negative (missed true match) "
      "silently leaves duplicate entities unmerged with no natural trigger for anyone to notice. This "
      "is a documented starting assumption, not a measured business figure -- override with "
      "`--cost-fp`/`--cost-fn` once real costs are known.\n")

    a("## Threshold Selection\n")
    a(f"**Selected threshold (min business cost, validation set only): {best_cost_row['threshold']}**\n")
    a(f"- Precision: {best_cost_row['precision']}")
    a(f"- Recall: {best_cost_row['recall']}")
    a(f"- F1: {best_cost_row['f1']}")
    cm = best_cost_row["confusion_matrix"]
    a(f"- Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    a(f"- Business cost at this threshold: {best_cost_row['business_cost']:.1f}")
    a(f"- ROC-AUC: {fmt(best_cost_row['roc_auc'])}  PR-AUC: {fmt(best_cost_row['pr_auc'])}\n")
    a(f"**For comparison, Phase 5's own max-F1 threshold was {phase5_f1_threshold:.2f}** "
      f"(F1={phase5_f1_metrics['f1']}, cost=not what it optimized for). The two selection criteria "
      f"can legitimately pick different thresholds: max-F1 balances precision/recall equally, while "
      f"the cost criterion here weights false negatives {args.cost_fn / args.cost_fp:.1f}x more than "
      f"false positives, which — given this task's asymmetric error costs — is arguably the more "
      f"appropriate criterion for a production decision, not just a diagnostic one.\n")

    if tuning_result:
        a("## Optional lightweight hyperparameter tuning\n")
        if tuning_result.get("ran"):
            a(f"- Ran: yes ({tuning_result['seconds']}s, {TUNE_N_ITER} candidates, "
              f"{TUNE_CV}-fold CV, scoring=average_precision, fit on X_train only)")
            a(f"- Best params: `{tuning_result['best_params']}`")
            a(f"- Best CV PR-AUC (train folds only): {tuning_result['best_cv_pr_auc']}")
            a(f"- Tuned model validation ROC-AUC: {tuning_result['val_roc_auc']}, "
              f"PR-AUC: {tuning_result['val_pr_auc']}")
            tcm = tuning_result["best_cost_row"]["confusion_matrix"]
            a(f"- Tuned model at its own min-cost threshold "
              f"({tuning_result['best_cost_row']['threshold']}): "
              f"precision={tuning_result['best_cost_row']['precision']}, "
              f"recall={tuning_result['best_cost_row']['recall']}, "
              f"f1={tuning_result['best_cost_row']['f1']}, "
              f"cost={tuning_result['best_cost_row']['business_cost']:.1f}\n")
        else:
            a(f"- Ran: no — {tuning_result['reason']}\n")
    else:
        a("## Optional lightweight hyperparameter tuning\n")
        a("- Not requested this run (pass `--tune` to attempt it; it will only actually run if Phase 5 "
          "materially beats Phase 4's PR-AUC by at least "
          f"{TUNE_GATE_MARGIN}, per the Phase 6 spec's \"only if\" condition).\n")

    a("## Blocking recall analysis (Phase 2 recall recovery investigation)\n")
    if blocking_summary is None:
        a("_reports/blocking_stats.json not found — run `python src/evaluate_blocking.py` first to "
          "enable this analysis. No blocking-recall conclusions can be drawn without it._\n")
    else:
        a(f"Based on `reports/blocking_stats.json` "
          f"(from a run over {blocking_summary['run_entities_evaluated']} entities):\n")
        a(f"- Recall: `{blocking_summary['recall']}`")
        a("- Strategies ranked by true-pair contribution (not additive — a pair can be recovered by "
          "more than one strategy):")
        for strat, n in blocking_summary["ranked_strategies"]:
            a(f"  - `{strat}`: {n:,}")
        if blocking_summary["low_value_strategies"]:
            a(f"- Low-value strategies (near-zero contribution in this run): "
              f"`{', '.join(blocking_summary['low_value_strategies'])}` — candidates for tightening "
              f"(reducing candidate volume) rather than relaxing, since relaxing a strategy that isn't "
              f"recovering true pairs would only add negative candidates.")
        else:
            a("- No strategy showed near-zero contribution in this run — no strategy stood out as "
              "obviously safe to remove on this data.")
        a(f"- Oversized blocks flagged: {blocking_summary['n_s2_problematic_blocks']} in S2, "
          f"{blocking_summary['n_s3_problematic_blocks']} in S3.")
        a("- **Important distinction:** the recall numbers above are BLOCKING recall (Phase 2) — the "
          "fraction of true matches that ever became a candidate. This is separate from MODEL recall "
          "(Phase 4/5, reported above) — the fraction of candidates the classifier itself correctly "
          "flags. A classifier can only ever recover a pair blocking already surfaced; the ceiling on "
          "achievable end-to-end recall is `blocking_recall × model_recall_among_candidates`, not "
          "model recall alone.\n")
        a("- **Per-entity miss attribution (which specific true matches were missed by which "
          "strategy) cannot be determined from `blocking_stats.json` alone** — it records aggregate "
          "counts, not which individual ground-truth pairs were missed. Determining that would require "
          "additional instrumentation in `evaluate_blocking.py` to log entity-level misses, which was "
          "deliberately NOT added here to avoid rewriting Phase 2 code beyond what this investigation "
          "requires; it is flagged below as a concrete next step.\n")

    if relax_result:
        a("## Targeted blocking relaxation experiment\n")
        b, r = relax_result["baseline"], relax_result["relaxed"]
        a(f"- Baseline (current Phase 2 defaults): recall={b['recall']}, "
          f"found {b['found_pairs']:,}/{b['gt_pairs']:,} true pairs, "
          f"candidate-count stats={b['candidate_count_stats']}")
        a(f"- Relaxed ({r['label']}): recall={r['recall']}, "
          f"found {r['found_pairs']:,}/{r['gt_pairs']:,} true pairs, "
          f"candidate-count stats={r['candidate_count_stats']}")
        if b["recall"] is not None and r["recall"] is not None:
            recall_delta = round(r["recall"] - b["recall"], 4)
            mean_b = b["candidate_count_stats"].get("mean", 0)
            mean_r = r["candidate_count_stats"].get("mean", 0)
            volume_delta_pct = round((mean_r - mean_b) / mean_b * 100, 1) if mean_b else None
            a(f"- **Recall delta: {recall_delta:+}. Mean candidate-volume delta: "
              f"{volume_delta_pct:+}%** (if positive, the relaxation recovered more true pairs at the "
              f"cost of more candidates to score; whether that tradeoff is worthwhile depends on the "
              f"cost of scoring more candidates versus the value of the extra recall recovered).\n")
    else:
        a("## Targeted blocking relaxation experiment\n")
        a("- Not run this time (pass `--relax-experiment`, with `dataset/` present, to try it). Per "
          "the Phase 6 spec, this is only worth running if the blocking-recall analysis above shows "
          "clear room for improvement — it is not run unconditionally.\n")

    a("## Comparison against Phase 4 and Phase 5\n")
    a(f"- Phase 4 baseline PR-AUC: {fmt(baseline_pr_auc)}")
    a(f"- Phase 5 PR-AUC: {fmt(phase5_default['pr_auc'])}")
    a(f"- Phase 6 selected threshold ({best_cost_row['threshold']}) vs Phase 5's own max-F1 threshold "
      f"({phase5_f1_threshold:.2f}): "
      f"F1 {best_cost_row['f1']} vs {phase5_f1_metrics['f1']}, "
      f"recall {best_cost_row['recall']} vs {phase5_f1_metrics['recall']}, "
      f"precision {best_cost_row['precision']} vs {phase5_f1_metrics['precision']}.")
    a("- Phase 6 does not change the underlying model from Phase 5 (unless `--tune` both ran and "
      "improved PR-AUC) — its contribution is choosing a threshold aligned to actual error costs "
      "instead of F1 alone, plus a documented, data-grounded read on whether blocking (not modeling) "
      "is the binding constraint on further recall gains.\n")

    a("## Limitations\n")
    a("- The 1:10 false-negative:false-positive cost ratio is a documented placeholder assumption, "
      "not a measured business figure; the selected threshold should be revisited once real costs "
      "are supplied via `--cost-fp`/`--cost-fn`.")
    a("- With positives this rare, small changes in a handful of predictions can shift precision/recall "
      "substantially — treat all point estimates here (Phase 6 included) as noisy.")
    a("- Blocking-recall figures (if present) come from whatever sample size "
      "`evaluate_blocking.py` was last run with — they are not necessarily representative of the "
      "full dataset's recall unless that run used `--full`.")
    a("- Per-entity miss attribution by blocking strategy is not currently possible from existing "
      "aggregate stats (see above) — a concrete next step for a future phase, not implemented here.")
    a("- This is a single stratified split; none of the comparisons here establish behavior across "
      "other seeds/folds.\n")

    a("## Conclusion\n")
    a(f"Phase 6 selects threshold **{best_cost_row['threshold']}** for the Phase 5 model under the "
      f"documented {args.cost_fp}:{args.cost_fn} (FP:FN) cost assumption, yielding "
      f"F1={best_cost_row['f1']}, precision={best_cost_row['precision']}, "
      f"recall={best_cost_row['recall']}, at business cost {best_cost_row['business_cost']:.1f} on "
      f"this validation split. "
      + ("Tuning was attempted and " + ("improved" if tuning_result and tuning_result.get("ran") and
         tuning_result.get("val_pr_auc", 0) and tuning_result["val_pr_auc"] > phase5_default["pr_auc"]
         else "did not clearly improve") + " on the untuned Phase 5 model. "
         if tuning_result and tuning_result.get("ran") else "Tuning was not run this time. ")
      + ("Blocking-recall analysis found data to investigate further; see the relaxation-experiment "
         "section above for the measured tradeoff." if relax_result else
         "No blocking relaxation experiment was run this time; run with --relax-experiment once the "
         "blocking-recall analysis above indicates it's warranted."))

    return "\n".join(lines)


if __name__ == "__main__":
    main()
