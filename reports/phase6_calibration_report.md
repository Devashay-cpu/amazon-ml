# Phase 6 Calibration Report

Generated: 2026-09-27 09:11:02

## Phase 5 baseline metrics (this run's split)

- Input: `D:\amazon-ml\student_resource\features\train_pair_features.csv`, 1,199,128 rows, 16 features
- Train/val split: 959,302 / 239,826 (seed=42, test_size=0.2)
- Phase 4 baseline: ROC-AUC=0.9999, PR-AUC=0.573
- Phase 5 model @ default 0.50: {'threshold': 0.5, 'accuracy': 0.9999, 'precision': 0.6667, 'recall': 0.8814, 'f1': 0.7591, 'confusion_matrix': {'tn': 239741, 'fp': 26, 'fn': 7, 'tp': 52}, 'roc_auc': 0.983, 'pr_auc': 0.877}
- Phase 5 model @ its own max-F1 threshold (0.96): {'threshold': np.float64(0.9600000000000001), 'accuracy': 0.9999, 'precision': 0.7612, 'recall': 0.8644, 'f1': 0.8095, 'confusion_matrix': {'tn': 239751, 'fp': 16, 'fn': 8, 'tp': 51}, 'roc_auc': 0.983, 'pr_auc': 0.877}

## Threshold analysis table (excerpt — every 10th of 99 grid points)

| threshold | precision | recall | f1 | fp | fn | business_cost |
|---|---|---|---|---|---|---|
| 0.01 | 0.534 | 0.9322 | 0.679 | 48 | 4 | 88.0 |
| 0.11 | 0.6395 | 0.9322 | 0.7586 | 31 | 4 | 71.0 |
| 0.21 | 0.6429 | 0.9153 | 0.7552 | 30 | 5 | 80.0 |
| 0.31 | 0.6585 | 0.9153 | 0.766 | 28 | 5 | 78.0 |
| 0.41 | 0.6667 | 0.9153 | 0.7714 | 27 | 5 | 77.0 |
| 0.51 | 0.6667 | 0.8814 | 0.7591 | 26 | 7 | 96.0 |
| 0.61 | 0.6842 | 0.8814 | 0.7704 | 24 | 7 | 94.0 |
| 0.71 | 0.6933 | 0.8814 | 0.7761 | 23 | 7 | 93.0 |
| 0.81 | 0.7123 | 0.8814 | 0.7879 | 21 | 7 | 91.0 |
| 0.91 | 0.7222 | 0.8814 | 0.7939 | 20 | 7 | 90.0 |

## Business cost definition

`cost = 1.0 * false_positives + 10.0 * false_negatives`

- cost_fp = 1.0, cost_fn = 10.0 (defaults)
- Rationale for the default 1:10 ratio: a false positive (incorrect merge) is a candidate pair a downstream review process can still catch and reverse; a false negative (missed true match) silently leaves duplicate entities unmerged with no natural trigger for anyone to notice. This is a documented starting assumption, not a measured business figure -- override with `--cost-fp`/`--cost-fn` once real costs are known.

## Threshold Selection

**Selected threshold (min business cost, validation set only): 0.11**

- Precision: 0.6395
- Recall: 0.9322
- F1: 0.7586
- Confusion matrix: TN=239736, FP=31, FN=4, TP=55
- Business cost at this threshold: 71.0
- ROC-AUC: 0.983  PR-AUC: 0.877

**For comparison, Phase 5's own max-F1 threshold was 0.96** (F1=0.8095, cost=not what it optimized for). The two selection criteria can legitimately pick different thresholds: max-F1 balances precision/recall equally, while the cost criterion here weights false negatives 10.0x more than false positives, which — given this task's asymmetric error costs — is arguably the more appropriate criterion for a production decision, not just a diagnostic one.

## Optional lightweight hyperparameter tuning

- Not requested this run (pass `--tune` to attempt it; it will only actually run if Phase 5 materially beats Phase 4's PR-AUC by at least 0.05, per the Phase 6 spec's "only if" condition).

## Blocking recall analysis (Phase 2 recall recovery investigation)

Based on `reports/blocking_stats.json` (from a run over 20000 entities):

- Recall: `{'overall': 0.9131, 's2': 0.9105, 's3': 0.9155, 'only_s2_entities': 0.9131, 'only_s3_entities': 0.9163, 'both_s2_s3_entities': 0.913}`
- Strategies ranked by true-pair contribution (not additive — a pair can be recovered by more than one strategy):
  - `country_name_prefix4`: 52,951
  - `address_alnum_prefix8`: 30,744
  - `country_suffix_stripped_name`: 27,527
  - `house_number_street`: 23,601
  - `country_exact_name`: 15,181
  - `postal_code`: 12,282
  - `country_rare_name_token`: 12,211
- No strategy showed near-zero contribution in this run — no strategy stood out as obviously safe to remove on this data.
- Oversized blocks flagged: 20 in S2, 20 in S3.
- **Important distinction:** the recall numbers above are BLOCKING recall (Phase 2) — the fraction of true matches that ever became a candidate. This is separate from MODEL recall (Phase 4/5, reported above) — the fraction of candidates the classifier itself correctly flags. A classifier can only ever recover a pair blocking already surfaced; the ceiling on achievable end-to-end recall is `blocking_recall × model_recall_among_candidates`, not model recall alone.

- **Per-entity miss attribution (which specific true matches were missed by which strategy) cannot be determined from `blocking_stats.json` alone** — it records aggregate counts, not which individual ground-truth pairs were missed. Determining that would require additional instrumentation in `evaluate_blocking.py` to log entity-level misses, which was deliberately NOT added here to avoid rewriting Phase 2 code beyond what this investigation requires; it is flagged below as a concrete next step.

## Targeted blocking relaxation experiment

- Not run this time (pass `--relax-experiment`, with `dataset/` present, to try it). Per the Phase 6 spec, this is only worth running if the blocking-recall analysis above shows clear room for improvement — it is not run unconditionally.

## Comparison against Phase 4 and Phase 5

- Phase 4 baseline PR-AUC: 0.573
- Phase 5 PR-AUC: 0.877
- Phase 6 selected threshold (0.11) vs Phase 5's own max-F1 threshold (0.96): F1 0.7586 vs 0.8095, recall 0.9322 vs 0.8644, precision 0.6395 vs 0.7612.
- Phase 6 does not change the underlying model from Phase 5 (unless `--tune` both ran and improved PR-AUC) — its contribution is choosing a threshold aligned to actual error costs instead of F1 alone, plus a documented, data-grounded read on whether blocking (not modeling) is the binding constraint on further recall gains.

## Limitations

- The 1:10 false-negative:false-positive cost ratio is a documented placeholder assumption, not a measured business figure; the selected threshold should be revisited once real costs are supplied via `--cost-fp`/`--cost-fn`.
- With positives this rare, small changes in a handful of predictions can shift precision/recall substantially — treat all point estimates here (Phase 6 included) as noisy.
- Blocking-recall figures (if present) come from whatever sample size `evaluate_blocking.py` was last run with — they are not necessarily representative of the full dataset's recall unless that run used `--full`.
- Per-entity miss attribution by blocking strategy is not currently possible from existing aggregate stats (see above) — a concrete next step for a future phase, not implemented here.
- This is a single stratified split; none of the comparisons here establish behavior across other seeds/folds.

## Conclusion

Phase 6 selects threshold **0.11** for the Phase 5 model under the documented 1.0:10.0 (FP:FN) cost assumption, yielding F1=0.7586, precision=0.6395, recall=0.9322, at business cost 71.0 on this validation split. Tuning was not run this time. No blocking relaxation experiment was run this time; run with --relax-experiment once the blocking-recall analysis above indicates it's warranted.