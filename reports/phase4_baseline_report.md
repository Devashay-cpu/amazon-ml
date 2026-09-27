# Phase 4 Baseline Classifier

Generated: 2026-09-26 23:43:25

## Dataset

- Input file: `D:\amazon-ml\student_resource\features\train_pair_features.csv`
- Rows loaded: 1,199,128
- Rows dropped (missing/invalid label): 0
- Rows used: 1,199,128
- Feature count: 16 (`name_exact_match, name_suffix_stripped_match, name_alnum_match, name_token_sorted_match, name_prefix4_match, name_char_similarity, name_token_jaccard, name_token_count_diff, address_exact_match, address_alnum_match, address_char_similarity, house_number_match, postal_code_match, unit_token_match, street_token_jaccard, country_match`)
- Identifier columns excluded from features: `s1_entity_id, candidate_entity_id, source`

## Class Distribution

- Positive count: 297 (0.02%)
- Negative count: 1,198,831 (99.98%)

## Model

- Model used: `LogisticRegression(class_weight="balanced")`
- Preprocessing: `SimpleImputer(missing_values=-1, strategy="median", add_indicator=True)` -> `StandardScaler()`, inside a single `sklearn.pipeline.Pipeline`
- Class imbalance handling: `class_weight="balanced"` (inverse-frequency reweighting), no oversampling/undersampling introduced
- Random seed: 42
- Train/validation split: 959302 / 239826 rows (test_size=0.2, stratified)
- Fit time: 10.979s

## Missing Sentinel Handling

Phase 3's `MISSING = -1` sentinel means "this comparison could not be made" (e.g. no postal code was parsed on one or both sides) -- it is NOT a normal similarity value and must never be treated as "0 = confirmed non-match" by the model. Handling used here:

1. `SimpleImputer(missing_values=-1, ...)` is told explicitly that `-1` (not `NaN`) marks missing values for these columns.
2. Each `-1` is replaced by that feature's **median computed only from the non-sentinel values in the training split** (fit on `X_train` only, applied unchanged to `X_val` -- no leakage).
3. `add_indicator=True` appends one extra binary column per feature that had at least one sentinel in training data, so the model can separately learn "this comparison was impossible" as its own signal, distinct from whatever the imputed value happens to be.
4. Phase 3's stored CSV values were not modified — the sentinel is only reinterpreted inside this Pipeline at train/predict time.

Missing-sentinel counts observed in the loaded data (feature: count):

- `name_exact_match`: 63,318
- `name_suffix_stripped_match`: 64,127
- `name_alnum_match`: 63,318
- `name_token_sorted_match`: 63,318
- `name_prefix4_match`: 63,318
- `name_char_similarity`: 63,318
- `name_token_jaccard`: 63,318
- `address_exact_match`: 28,848
- `address_alnum_match`: 28,848
- `address_char_similarity`: 28,848
- `house_number_match`: 752,898
- `postal_code_match`: 719,970
- `unit_token_match`: 765,380
- `street_token_jaccard`: 29,238

## Validation Results

### At default threshold (0.50)

- Accuracy: 0.9943
- Precision: 0.0416
- Recall: 1.0
- F1: 0.0798
- ROC-AUC: 0.9999
- PR-AUC: 0.6276
- Confusion matrix: TN=238407, FP=1360, FN=0, TP=59

## Threshold Selection

**Criterion:** maximize F1 over a 0.01-step grid of thresholds from 0.01 to 0.99, evaluated ONLY on the validation set (never on a held-out test set, since none was tuned against here). F1 is used because entity matching needs precision and recall balanced jointly, per spec, and F1 is the standard, interpretable way to balance them for a baseline -- not a business-specific cost model. Ties are broken by choosing the threshold closest to 0.50.

- Default (0.50) F1: 0.0798
- Selected threshold: 0.99
- Selected-threshold Accuracy: 0.9988
- Selected-threshold Precision: 0.1681
- Selected-threshold Recall: 0.9831
- Selected-threshold F1: 0.2871
- Selected-threshold confusion matrix: TN=239480, FP=287, FN=1, TP=58

## Limitations

- This is a baseline model (Logistic Regression) intended to establish a floor for how well the Phase 3 engineered features separate matches from non-matches -- not a production classifier, and hyperparameters were not extensively tuned.
- Achievable recall is capped by Phase 2's blocking recall: a true match that blocking never produced as a candidate cannot appear in this dataset at all, so no classifier trained on it can recover that pair. This model's recall reflects performance only on the subset of true matches that blocking actually surfaced as candidates.
- Validation performance here is measured on a single stratified split of the current feature-generation run and does not represent final production performance, especially if this was run on a small `--sample-size` rather than the full dataset.
- Missing-sentinel rates (above) reflect the heuristic, country-agnostic postal-code/house-number extraction from Phase 2/3; features with high missingness contribute mostly through their indicator columns rather than their imputed values.