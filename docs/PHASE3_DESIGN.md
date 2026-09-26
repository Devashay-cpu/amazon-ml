# Phase 3 Design — Candidate-Pair Feature Engineering

## Where this fits

```
Raw data
  -> normalization.py            (Phase 1/2, unchanged)
  -> blocking.py (BlockIndex)    (Phase 2, unchanged)
  -> candidate_generation.py     (Phase 2, unchanged: build_index_from_source, iter_candidates
                                   Phase 3 addition: build_record_lookup / build_record_lookup_for_ids)
  -> features.py                 (Phase 3, new: extract_pair_features)
  -> build_feature_dataset.py     (Phase 3, new: driver — pairs + features + labels -> CSV)
```

## Fields used (no invented fields)

Only the fields that exist in the dataset are used:
- Source files (`entity_id`, `business_name`, `business_address`, `country`) via
  `normalize_business_name` / `normalize_address` / `normalize_country`.
- Ground truth (`source1_entity_id`, `matched_entity_ids`) via the existing `gt_loader.load_ground_truth`.

No city/state/phone/etc. fields are referenced because they don't exist in this dataset.

## Feature definitions (16 features, `src/features.py`)

| Feature | Definition | Why |
|---|---|---|
| `name_exact_match` | 1/0/MISSING — `exact_normalized` strings equal | Catches identical names after case/punctuation cleanup |
| `name_suffix_stripped_match` | 1/0/MISSING — `suffix_stripped` equal | Catches "Acme Corp" vs "Acme Corporation" (legal-suffix noise) |
| `name_alnum_match` | 1/0/MISSING — `alnum_normalized` equal | Robust to spacing/punctuation differences entirely |
| `name_token_sorted_match` | 1/0/MISSING — `token_sorted` equal | Catches word-order swaps ("Sons Sharma" vs "Sharma Sons") |
| `name_prefix4_match` | 1/0/MISSING — first 4 alnum chars equal | Cheap proxy consistent with the blocking key of the same name |
| `name_char_similarity` | difflib ratio on `exact_normalized`, 0-1 | Continuous signal for near-miss spelling/typo differences |
| `name_token_jaccard` | set-overlap of name tokens, 0-1 | Continuous signal robust to extra/missing words |
| `name_token_count_diff` | absolute difference in token counts | Flags very different name lengths/structures |
| `address_exact_match` | 1/0/MISSING — `address_normalized` equal | Same-address confirmation |
| `address_alnum_match` | 1/0/MISSING — `address_alnum` equal | Robust to formatting-only differences |
| `address_char_similarity` | difflib ratio on `address_normalized`, 0-1 | Continuous signal for near-miss addresses |
| `house_number_match` | 1/0/MISSING — `house_number` equal | Strong discriminating signal when present |
| `postal_code_match` | 1/0/MISSING — `postal_code` equal | Strong discriminating signal when present, country-agnostic extraction |
| `unit_token_match` | 1/0/MISSING — `unit_token` equal | Distinguishes different suites/units at the same street address |
| `street_token_jaccard` | set-overlap of `street_tokens`, 0-1 | Continuous street-name similarity independent of house number/postal code |
| `country_match` | 1/0/MISSING — `normalized` country equal | Sanity check (blocking already conditions on country for most strategies, so this is near-constant 1 for most candidates, but not guaranteed for every strategy, e.g. `postal_code`) |

All 16 are pure functions of the two records' normalized fields — deterministic, no I/O, no randomness.

## Missing-value handling

Every feature uses **`MISSING = -1`** (not `0`) when either side lacks the underlying value (e.g.
no postal code parsed from the address). This is a data-quality decision: collapsing "we don't
know" into "confirmed non-match" (`0`) would silently bias any downstream classifier. `0` always
means "compared and found different"; `-1` always means "could not compare." The Phase 3 report
(`reports/phase3_feature_report.md`) prints missing-value rates per feature per run so this is
visible, not hidden.

Two blank records never produce a spurious match: `test_both_missing_is_not_reported_as_a_confident_match`
guards this explicitly.

## Input / output format

**Input** (per pair, internal): two dicts shaped like
`{"name": normalize_business_name(...), "address": normalize_address(...), "country": normalize_country(...)}`
— one for the Source-1 record, one for the S2/S3 candidate record.

**Output** (`features/train_pair_features.csv` or `features/test_pair_features.csv`), one row per
candidate pair, columns:

```
s1_entity_id, candidate_entity_id, source, <16 feature columns>, label
```

- `source` is `S2` or `S3` (needed because S2/S3 IDs are not guaranteed globally unique).
- `label` is `1` (true match), `0` (non-match), or empty (`test` dataset — no ground truth exists).

## Label generation

For each `(s1_entity_id, source, candidate_entity_id)` triple that blocking actually produced as a
candidate, the label is `1` if `candidate_entity_id` is in that S1 entity's ground-truth matched set
for that source (via the unmodified `gt_loader.load_ground_truth`), else `0`. Ground truth is
**never** used as a feature input — only to assign the label column after features are already
computed and fixed — so there is no label leakage into the feature vector.

Duplicate `(s1, source, candidate)` triples are deduplicated (first occurrence kept, later ones
counted and reported, not silently dropped or double-counted).

## Limitations

1. **Recall ceiling inherited from Phase 2.** A true match that Phase 2's blocking never generated
   as a candidate cannot appear as a positive row here — it's simply absent, not mislabeled `0`.
   Whatever recall ceiling Phase 2 measured on a given run (91.31% overall on the reported 20,000-entity
   validation) upper-bounds what any Phase 4 classifier can recover on that same run, unless blocking
   is revisited.
2. **Postal-code / house-number extraction is heuristic**, not a true parser for any specific
   country's address format (deliberately, per the country-agnostic requirement from Phase 2). Some
   legitimate matches may show `MISSING` for these fields, which the classifier should treat as
   "no signal," not "non-match" (see missing-value handling above).
3. **Class imbalance is expected** (most blocked candidates are non-matches) and is a Phase 4
   modeling concern (e.g. class weights, resampling), not something this phase corrects.
4. **`build_record_lookup_for_ids` still loads full normalized records for every candidate ID into
   memory, per batch (see "Known OOM fix" below).** The batch-size control bounds how many are held
   at once; if a single batch is still too large for available memory, the documented next step is a
   disk-backed lookup (e.g. SQLite keyed by entity_id) rather than a change to the feature logic itself.

## Known OOM fix (2026-09-26)

**Symptom:** `python src/build_feature_dataset.py --sample-size 20000` failed with
`pandas.errors.ParserError: Error tokenizing data. C error: out of memory` inside
`build_record_lookup_for_ids`, after S2/S3 index construction, ground-truth loading, and candidate
scanning all completed successfully.

**Root cause:** the driver script (not `candidate_generation.py` or `blocking.py`) used to scan
*all* requested S1 entities first, accumulating one global `needed_s2_ids` / `needed_s3_ids` set and
one global pair index for the entire run, and only then loaded records for the full union. At 20,000
S1 entities that union reached ~3.65M / ~3.85M ids — roughly 73% of each 5M+ record source — so by
the time `pd.read_csv` was called to load those records, the process had already exhausted memory
holding the two `BlockIndex` objects, the global ID sets, and the global pair index; the C parser's
own read-buffer allocation failed mid-tokenize. `build_record_lookup_for_ids`'s own chunked-read-and-
filter implementation was already correct in isolation — the bug was entirely in how much the caller
asked it to hold onto **at once**.

**Fix (control-flow only; no feature, blocking, or schema changes):** `build_feature_dataset.py` now
streams S1 entities in bounded batches (`--batch-size`, default 2000). For each batch: compute needed
ids for just that batch, load only those records via the unchanged `build_record_lookup_for_ids`,
extract features and write rows immediately, then explicitly release the batch's ID sets and record
dicts before starting the next batch. Peak memory is now bounded by one batch's candidate records
instead of the whole run's. The duplicate-pair guard was correspondingly rescoped to per-batch (from
global), which is safe because each S1 entity is visited exactly once across the whole run — no two
batches can ever produce the same `(s1_id, source, candidate_id)` triple.

**Tradeoff:** S2/S3 are now re-scanned once per batch instead of once total — more disk I/O, bounded
memory. Use a larger `--batch-size` for speed if memory allows; a smaller one if it doesn't.

**Verified unchanged:** `tests/test_build_feature_dataset.py` runs the same toy dataset through
`--batch-size 1` and `--batch-size 1000` and asserts byte-identical output rows, confirming the fix
changed only memory behavior, not results.

## How to run

```
python src/build_feature_dataset.py                        # train, first 20,000 S1 rows (default)
python src/build_feature_dataset.py --sample-size 100000    # bigger validation sample
python src/build_feature_dataset.py --full                  # entire train_source1.tsv
python src/build_feature_dataset.py --dataset test          # test set, features only, no labels
```

Run the tests:
```
python tests/test_normalization.py           # Phase 2, unchanged, still passing
python tests/test_blocking.py                # Phase 2, unchanged, still passing
python tests/test_features.py                # Phase 3, unchanged, still passing
python tests/test_build_feature_dataset.py   # Phase 3, new — memory-fix regression tests
```
