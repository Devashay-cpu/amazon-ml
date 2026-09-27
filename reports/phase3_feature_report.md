# Phase 3 Feature Report

Generated: 2026-09-26 23:02:51

- Dataset: `train`
- S1 entities processed: 100 (first 100)
- Batch size used: 2 (memory-fix control; see module docstring)
- Largest single batch's combined needed-ID count: 162,304 (peak memory is bounded by this, not by the run's total distinct candidate ids)
- Candidate pair rows written: 1,199,128
- Duplicate candidate pairs skipped (already seen): 0
- Candidate ids present in blocking output but unresolvable in source file (data-quality flag, should be 0): 0
- Output file: `features\train_pair_features.csv`

## Label / class balance

- Positive pairs (label=1): 297
- Negative pairs (label=0): 1,198,831
- Positive rate: 0.0002
- Class imbalance is expected and typical for candidate-pair datasets in entity resolution (most blocked candidates are non-matches); this is a Phase 4 modeling consideration (e.g. class weighting or resampling), not a Phase 3 bug.

## Missing-value counts per feature (sentinel = -1, meaning 'no data', not 'non-match')

- `name_exact_match`: 63,318 (5.28%)
- `name_suffix_stripped_match`: 64,127 (5.35%)
- `name_alnum_match`: 63,318 (5.28%)
- `name_token_sorted_match`: 63,318 (5.28%)
- `name_prefix4_match`: 63,318 (5.28%)
- `name_char_similarity`: 63,318 (5.28%)
- `name_token_jaccard`: 63,318 (5.28%)
- `name_token_count_diff`: 0 (0.0%)
- `address_exact_match`: 28,848 (2.41%)
- `address_alnum_match`: 28,848 (2.41%)
- `address_char_similarity`: 28,848 (2.41%)
- `house_number_match`: 752,898 (62.79%)
- `postal_code_match`: 719,970 (60.04%)
- `unit_token_match`: 765,380 (63.83%)
- `street_token_jaccard`: 29,238 (2.44%)
- `country_match`: 0 (0.0%)

## Label-leakage check

- Features are computed exclusively from `normalize_business_name` / `normalize_address` / `normalize_country` outputs on the S1 record and the candidate record. Ground truth is used ONLY to assign the `label` column after features are already fixed, never as a feature input, so no ground-truth information leaks into the feature vector itself.
- `source1_entity_id` / `candidate_entity_id` are kept in the output for traceability but are raw dataset IDs, not derived from any label — they carry no leakage risk on their own, but should not be used as model features directly (arbitrary IDs have no generalizable signal).

## Known limitation (inherent to the two-stage blocking + classification design)

- This dataset can only contain a positive pair if blocking actually produced that candidate. Phase 2 measured blocking recall on this same run's entities; any true match blocking missed is absent here as a positive row (it simply never appears as a candidate), not mislabeled as negative. Phase 4's achievable recall is therefore capped by Phase 2's blocking recall — improving it later means revisiting blocking, not the classifier.

## Memory-fix note (this run)

- Processed in batches of 2 S1 entities. The largest single batch needed 162,304 candidate ids resolved at once, versus the run's full distinct-id total which would be much larger if computed globally — this bound is what avoids the previous out-of-memory failure. If this run still uses too much memory, re-run with a smaller `--batch-size`; if it finishes with room to spare, a larger `--batch-size` will finish faster (fewer S2/S3 re-scans) at the cost of higher peak memory.
