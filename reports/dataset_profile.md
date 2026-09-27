# Dataset Profile — Business Entity Resolution (Phase 1)

Generated: 2026-09-25 15:07:11  
Project root: `D:\amazon-ml\student_resource`  
Chunk size: 100000  
Random seed: 42

## A. Dataset overview

- Train files found: 4
- Test files found: 3

## B. File sizes and row counts

| Split | File | Size | Rows | Columns |
|---|---|---|---|---|
| train | train_ground_truth.tsv | 121.13 MB | 2,206,821 | 2 |
| train | train_source1.tsv | 200.34 MB | 2,206,821 | 4 |
| train | train_source2.tsv | 466.63 MB | 5,034,616 | 4 |
| train | train_source3.tsv | 480.37 MB | 5,285,603 | 4 |
| test | test_source1.tsv | 166.91 MB | 1,732,544 | 4 |
| test | test_source2.tsv | 485.86 MB | 4,887,273 | 4 |
| test | test_source3.tsv | 482.56 MB | 5,082,316 | 4 |

## C. Column analysis

**train/train_ground_truth.tsv**: `source1_entity_id, matched_entity_ids`  
Detected — id: `source1_entity_id`, name: `None`, address: `None`, country: `None`

**train/train_source1.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

**train/train_source2.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

**train/train_source3.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

**test/test_source1.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

**test/test_source2.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

**test/test_source3.tsv**: `entity_id, business_name, business_address, country`  
Detected — id: `entity_id`, name: `business_name`, address: `business_address`, country: `country`

## D. Missing-value analysis

**train/train_ground_truth.tsv**
- `matched_entity_ids`: 123,247 missing (5.58%)

**train/train_source1.tsv**

**train/train_source2.tsv**
- `business_name`: 2 missing (0.00%)
- `business_address`: 168,967 missing (3.36%)

**train/train_source3.tsv**
- `business_name`: 13 missing (0.00%)
- `business_address`: 175,916 missing (3.33%)

**test/test_source1.tsv**

**test/test_source2.tsv**
- `business_name`: 46 missing (0.00%)
- `business_address`: 129,408 missing (2.65%)

**test/test_source3.tsv**
- `business_name`: 59 missing (0.00%)
- `business_address`: 136,098 missing (2.68%)

## E. Country distribution

**train/train_source1.tsv** — 2 unique countries seen
- US: 1,323,633
- India: 883,188

**train/train_source2.tsv** — 2 unique countries seen
- US: 3,016,817
- India: 2,017,799

**train/train_source3.tsv** — 2 unique countries seen
- US: 3,170,056
- India: 2,115,547

**test/test_source1.tsv** — 3 unique countries seen
- India: 809,986
- US: 663,106
- France: 259,452

**test/test_source2.tsv** — 3 unique countries seen
- India: 2,312,565
- US: 1,871,330
- France: 703,378

**test/test_source3.tsv** — 3 unique countries seen
- India: 2,405,000
- US: 1,945,701
- France: 731,615

## F. Name statistics (business_name length, characters)

**train/train_source1.tsv**: mean=24.03, std=7.74, min=3, max=105, percentiles={'p25': 18, 'p50': 24, 'p75': 30, 'p90': 34, 'p99': 42}
**train/train_source2.tsv**: mean=25.1, std=8.89, min=2, max=104, percentiles={'p25': 18, 'p50': 25, 'p75': 31, 'p90': 37, 'p99': 47}
**train/train_source3.tsv**: mean=25.2, std=9.49, min=2, max=123, percentiles={'p25': 18, 'p50': 25, 'p75': 31, 'p90': 37, 'p99': 51}
**test/test_source1.tsv**: mean=23.84, std=7.67, min=3, max=92, percentiles={'p25': 18, 'p50': 24, 'p75': 29, 'p90': 34, 'p99': 42}
**test/test_source2.tsv**: mean=25.7, std=9.12, min=2, max=102, percentiles={'p25': 19, 'p50': 25, 'p75': 32, 'p90': 38, 'p99': 48}
**test/test_source3.tsv**: mean=25.66, std=9.57, min=2, max=103, percentiles={'p25': 19, 'p50': 25, 'p75': 32, 'p90': 38, 'p99': 51}

## G. Address statistics (business_address length, characters)

**train/train_source1.tsv**: mean=52.07, std=25.33, min=11, max=256, percentiles={'p25': 33, 'p50': 41, 'p75': 70, 'p90': 90, 'p99': 124}
**train/train_source2.tsv**: mean=47.83, std=23.7, min=8, max=249, percentiles={'p25': 31, 'p50': 37, 'p75': 62, 'p90': 84, 'p99': 119}
**train/train_source3.tsv**: mean=48.32, std=20.18, min=2, max=240, percentiles={'p25': 35, 'p50': 42, 'p75': 55, 'p90': 79, 'p99': 118}
**test/test_source1.tsv**: mean=57.21, std=25.03, min=11, max=268, percentiles={'p25': 36, 'p50': 51, 'p75': 75, 'p90': 93, 'p99': 126}
**test/test_source2.tsv**: mean=51.78, std=24.27, min=5, max=269, percentiles={'p25': 33, 'p50': 44, 'p75': 68, 'p90': 87, 'p99': 122}
**test/test_source3.tsv**: mean=50.08, std=21.44, min=5, max=267, percentiles={'p25': 36, 'p50': 44, 'p75': 59, 'p90': 81, 'p99': 119}

## H. Ground-truth match cardinality

- Total Source 1 entities: 2,206,821
- Zero-match entities: 2,206,821
- One-match entities: 0
- Multi-match entities: 0
- Max matches for a single S1 entity: 0 (example id: None)
- Match-count distribution: `{0: 2206821}`
- S2 match-count distribution: `{0: 2206821}`
- S3 match-count distribution: `{0: 2206821}`

## I. Singleton statistics

- Entities with only an S2 match: 0
- Entities with only an S3 match: 0
- Entities with both S2 and S3 matches: 0

> ⚠️ Could not confidently detect ground-truth ID/match columns from candidate name lists — inspect 'columns_raw' above and adjust GT_*_CANDIDATES at the top of this script to match the actual header names.

## J. Noise-pattern examples (from deterministic + random samples)

See `dataset_profile.json` → `samples` for the full head/random rows per file. Manually scan these for: abbreviations (Pvt/Ltd/Corp), punctuation differences, spelling variants, word-order changes, duplicated words, transliteration-like differences in names; Rd/Road, St/Street, missing components, reordering, landmark references, PIN/postal codes, and state/city variation in addresses.

## K. Important observations

_Fill in after reviewing the JSON output — auto-generated placeholder for manual notes._

## L. Risks for false positives

_Given precision-weighted F0.5, note here which noise patterns are most likely to cause false matches (e.g., common chain/franchise names, generic strip-mall addresses, near-duplicate names across different countries)._

## M. Recommendations for normalization

_To be filled in for Phase 2 based on the noise patterns observed above._

## N. Recommendations for blocking

_To be filled in for Phase 2 — e.g. candidate blocking keys such as normalized name prefix, postal code, city, or country, informed by the country distribution and address statistics above._
