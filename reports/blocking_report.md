# Blocking Report (Phase 2)

- Dataset: `train`
- Entities evaluated: 20,000 (first 20000)
- S2 index build time: 2273.35s
- S3 index build time: 3631.43s
- Candidate generation time: 144.39s
- Peak traced memory: 7905.5 MB

## Strategies tested

- `country_exact_name`
- `country_name_prefix4`
- `country_suffix_stripped_name`
- `country_rare_name_token`
- `postal_code`
- `house_number_street`
- `address_alnum_prefix8`

## Blocking recall

- Overall: **91.31%**
- S2: 91.05%
- S3: 91.55%
- Entities with only S2 matches: 91.31%
- Entities with only S3 matches: 91.63%
- Entities with both S2 and S3 matches: 91.30%

## Strategy contribution (true pairs recovered by each strategy)

Note: a pair can be recovered by more than one strategy; these are not additive.

- `country_name_prefix4`: 52,951 true pairs recovered
- `address_alnum_prefix8`: 30,744 true pairs recovered
- `country_suffix_stripped_name`: 27,527 true pairs recovered
- `house_number_street`: 23,601 true pairs recovered
- `country_exact_name`: 15,181 true pairs recovered
- `postal_code`: 12,282 true pairs recovered
- `country_rare_name_token`: 12,211 true pairs recovered

## Candidate-count statistics (per Source-1 entity)

**Total (S2+S3)** — mean=11068.76, median=5291.0, p90=22589, p95=48720, p99=96671, max=141040, min=0
**S2 only** — mean=5607.25, median=2404.0, p90=10995, p95=27067, p99=53826, max=83590, min=0
**S3 only** — mean=5461.51, median=2613.0, p90=11482, p95=22808, p99=43196, max=64948, min=0

## Problematic (oversized) blocks

**S2** — 20 block(s) over threshold:
- `postal_code` key=`india|2nd` size=53,269
- `postal_code` key=`india|1st` size=48,528
- `country_name_prefix4` key=`india|priv` size=41,490
- `postal_code` key=`india|3rd` size=37,174
- `country_name_prefix4` key=`united states|pedi` size=30,732
- `country_name_prefix4` key=`united states|inte` size=27,624
- `country_name_prefix4` key=`india|shri` size=26,764
- `address_alnum_prefix8` key=`india|maharash` size=26,367
- `postal_code` key=`india|4th` size=23,315
- `address_alnum_prefix8` key=`india|officeno` size=16,508

**S3** — 20 block(s) over threshold:
- `country_name_prefix4` key=`india|priv` size=49,720
- `postal_code` key=`india|2nd` size=41,930
- `postal_code` key=`india|1st` size=38,491
- `country_name_prefix4` key=`united states|pedi` size=31,261
- `country_name_prefix4` key=`india|shri` size=30,313
- `postal_code` key=`india|3rd` size=28,862
- `country_name_prefix4` key=`united states|inte` size=28,252
- `country_name_prefix4` key=`india|limi` size=19,771
- `postal_code` key=`india|4th` size=18,271
- `address_alnum_prefix8` key=`india|officeno` size=16,220

## Strategy key-space size (number of distinct blocking keys generated)

**S2**
- `country_exact_name`: 3,834,140 distinct keys
- `country_name_prefix4`: 196,352 distinct keys
- `country_suffix_stripped_name`: 3,306,176 distinct keys
- `country_rare_name_token`: 802,913 distinct keys
- `postal_code`: 85,791 distinct keys
- `house_number_street`: 1,640,321 distinct keys
- `address_alnum_prefix8`: 2,613,705 distinct keys
**S3**
- `country_exact_name`: 4,146,518 distinct keys
- `country_name_prefix4`: 200,613 distinct keys
- `country_suffix_stripped_name`: 3,612,076 distinct keys
- `country_rare_name_token`: 859,626 distinct keys
- `postal_code`: 89,966 distinct keys
- `house_number_street`: 1,678,799 distinct keys
- `address_alnum_prefix8`: 2,635,231 distinct keys

## Memory / runtime observations

- Peak traced Python memory during this run: 7905.5 MB.
- Both source files are read twice when building each index (pass 1: name-token frequency only; pass 2: full index) to avoid holding every normalized record in memory between passes — a deliberate CPU-for-memory tradeoff.
- No all-pairs comparison is performed anywhere; every candidate lookup is a dict hit against a precomputed inverted index.
- For the full ~2.2M-row training set, in-memory dict-of-lists indexes are expected to scale to tens of millions of keys; if memory becomes a constraint, the recommended next step is a disk-backed index (e.g. SQLite with an indexed (strategy, key) -> id table) rather than changing the blocking logic itself.

## Recommended final blocking strategy

Based on this run, the strategies recovering the most true pairs were: `country_name_prefix4`, `address_alnum_prefix8`, `country_suffix_stripped_name`. Recommend keeping the full strategy set for Phase 3 candidate generation (recall is the priority at this stage), while monitoring the 'problematic blocks' above as targets for future refinement (e.g. combining a rare-token key with a secondary signal) once precision-side tuning begins.