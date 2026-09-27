# Phase 5 Model Report

Generated: 2026-09-27 20:32:44

## Dataset

- Input file: `D:\amazon-ml\student_resource\features\train_pair_features.csv`
- Rows used: 1,199,128
- Feature count: 16 (`name_exact_match, name_suffix_stripped_match, name_alnum_match, name_token_sorted_match, name_prefix4_match, name_char_similarity, name_token_jaccard, name_token_count_diff, address_exact_match, address_alnum_match, address_char_similarity, house_number_match, postal_code_match, unit_token_match, street_token_jaccard, country_match`) -- unchanged from Phase 3/4

## Class Distribution

- Positive count: 297 (0.0248%)
- Negative count: 1,198,831 (99.9752%)
- Train/validation split: 959,302 / 239,826 rows (test_size=0.2, stratified, seed=42 -- identical methodology to Phase 4)

## Models Compared

### Phase 4 baseline (refit on this run's identical split)

- `LogisticRegression(class_weight="balanced")` inside a `SimpleImputer(missing_values=-1, add_indicator=True) -> StandardScaler` Pipeline (`train_baseline.build_pipeline`, imported unchanged).

### Phase 5 model

- `HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, max_iter=200)` -- scikit-learn's native histogram-based gradient boosting; no new dependency (xgboost/lightgbm were deliberately not introduced).
- Chosen over a plain `GradientBoostingClassifier` or a large `RandomForestClassifier` specifically for memory/practicality on ~1.2M rows: histogram binning keeps memory bounded regardless of row count, unlike a forest of many deep, unbinned trees.
- Random seed: 42 (same seed as the baseline run above)

## Missing Sentinel Handling

Phase 3's `MISSING = -1` sentinel means "this comparison could not be made" and must never be treated as a normal similarity value (unchanged principle from Phase 4). Handling differs by model family, not by redefinition of the sentinel itself:

- **Baseline (linear model):** `-1` is imputed to the training-split median plus a missing-indicator column (Phase 4's approach, reused unchanged here).
- **Phase 5 (tree model):** `-1` is converted to `np.nan` (a pure representation change, not a value change), and `HistGradientBoostingClassifier` natively learns, per split, which branch missing values should follow — using missingness as signal directly rather than through a separate indicator column.
- Neither approach modifies Phase 3's stored CSV values; both reinterpret the sentinel only inside their respective Pipelines at train/predict time.

## Validation Results

### Phase 4 baseline (LogisticRegression)

Fit time: 9.623s

**At default threshold (0.50):**
- Accuracy: 0.9943  Precision: 0.0413  Recall: 1.0  F1: 0.0794
- ROC-AUC: 0.9999  PR-AUC: 0.573
- Confusion matrix: TN=238399, FP=1368, FN=0, TP=59

**At selected threshold (0.99, max-F1 on validation set):**
- Accuracy: 0.9989  Precision: 0.1763  Recall: 0.9831  F1: 0.299
- Confusion matrix: TN=239496, FP=271, FN=1, TP=58

### Phase 5 model (HistGradientBoostingClassifier)

Fit time: 13.071s

**At default threshold (0.50):**
- Accuracy: 0.9999  Precision: 0.6667  Recall: 0.8814  F1: 0.7591
- ROC-AUC: 0.983  PR-AUC: 0.877
- Confusion matrix: TN=239741, FP=26, FN=7, TP=52

**At selected threshold (0.96, max-F1 on validation set):**
- Accuracy: 0.9999  Precision: 0.7612  Recall: 0.8644  F1: 0.8095
- Confusion matrix: TN=239751, FP=16, FN=8, TP=51

## Phase 5 vs Phase 4 Comparison (same split, selected threshold for each model)

- Precision: baseline=0.1763 -> phase5=0.7612 (+0.5849 (better))
- Recall: baseline=0.9831 -> phase5=0.8644 (-0.1187 (worse))
- F1: baseline=0.299 -> phase5=0.8095 (+0.5105 (better))
- ROC-AUC: baseline=0.9999 -> phase5=0.983 (-0.0169 (worse))
- PR-AUC: baseline=0.573 -> phase5=0.877 (+0.304 (better))

PR-AUC is the most informative single number here given the severity of the class imbalance (positive rate well under 5%) — ROC-AUC can look deceptively high under heavy imbalance because true negatives dominate.

## Limitations

- Neither model can recover a true match that Phase 2's blocking never produced as a candidate in the first place — the achievable recall ceiling from Phase 2/3 is unchanged and applies identically to both models here.
- `max_iter` for the Phase 5 model was fixed, not tuned via search, per the Phase 5 instruction to avoid expensive hyperparameter sweeps; further gains are plausible from modest tuning (e.g. `max_leaf_nodes`, `learning_rate`) but that is explicitly out of scope here.
- With positives this rare, validation metrics (especially precision) can be volatile — a handful of additional false positives/negatives can move precision/F1 substantially. Treat point estimates here as noisy, not exact.
- This comparison used a single stratified split; it does not by itself establish that Phase 5 generalizes better across other splits/seeds (that would need repeated/k-fold evaluation, out of scope for this baseline-vs-improved-model comparison).