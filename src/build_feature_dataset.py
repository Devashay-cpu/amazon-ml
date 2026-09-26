"""
Phase 3 — Build the ML-ready candidate-pair feature dataset

Pipeline (Phase 1/2 pieces reused UNCHANGED — see MEMORY FIX note below for
what changed and why):
    train_source2/3.tsv --(blocking.BlockIndex, unchanged)--> S2/S3 indexes
    train_source2/3.tsv --(candidate_generation.build_record_lookup_for_ids, unchanged)--> S2/S3 field lookups, per batch
    train_source1.tsv   --(local generator below, mirrors candidate_generation.iter_candidates)--> per-S1 candidate ID sets + normalized record
    for each (s1, candidate) pair --(features.extract_pair_features, unchanged)--> feature dict
    ground truth --(gt_loader.load_ground_truth, unchanged)--> label 1/0

Usage:
    python src/build_feature_dataset.py                      # train, first 20,000 S1 rows
    python src/build_feature_dataset.py --sample-size 100000
    python src/build_feature_dataset.py --full
    python src/build_feature_dataset.py --dataset test        # features only, no labels (test has no ground truth)
    python src/build_feature_dataset.py --batch-size 1000     # smaller batches = lower peak memory, more I/O

Writes:
    features/train_pair_features.csv  (or test_pair_features.csv)
    reports/phase3_feature_report.md

--------------------------------------------------------------------------
MEMORY FIX (see docs/PHASE3_DESIGN.md "Known OOM fix" section for detail):

Root cause of the earlier OOM: this script used to (1) stream ALL S1 rows
in the requested sample first, accumulating a SINGLE global
needed_s2_ids / needed_s3_ids set and a SINGLE global pair_index list for
every entity in the run, and only THEN call
build_record_lookup_for_ids(...) once for the entire unioned ID set. At
20,000 S1 entities that union reached ~3.65M / ~3.85M ids — roughly 73% of
each 5M+ record source — so by the time pandas.read_csv was called to load
those records, the process had already exhausted available memory holding
the global ID sets, the global pair index, and the two BlockIndex objects,
and the C parser's own read-buffer allocation failed mid-tokenize.

build_record_lookup_for_ids() itself was already correct (chunked read +
immediate filtering, never loading a full file) — the bug was entirely in
HOW MUCH it was ever asked to hold onto at once, driven by this script's
control flow, not by candidate_generation.py or blocking.py.

Fix: process S1 entities in bounded BATCHES (--batch-size, default 2000).
For each batch: compute needed ids for JUST that batch (a small fraction
of the global total), load only those records, extract features, write
rows immediately, then explicitly release the batch's ID sets and record
dicts before starting the next batch. Peak memory is now bounded by one
batch's worth of candidate records instead of the whole run's. This is a
control-flow change ONLY: no feature definition, no blocking strategy, and
no candidate-generation semantics changed — every candidate pair that
Phase 2's blocking would have produced is still produced and featurized
exactly the same way, just processed and flushed in smaller groups.
Tradeoff: S2/S3 are now re-scanned once per batch instead of once total,
trading more disk I/O for bounded memory (see docs/PHASE3_DESIGN.md).
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from candidate_generation import CHUNK_SIZE, build_index_from_source, build_record_lookup_for_ids
from features import FEATURE_NAMES, extract_pair_features
from gt_loader import load_ground_truth
from normalization import normalize_address, normalize_business_name, normalize_country

OUTPUT_COLUMNS = ["s1_entity_id", "candidate_entity_id", "source"] + FEATURE_NAMES + ["label"]
DEFAULT_BATCH_SIZE = 2000


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent
    if (candidate / "dataset").exists():
        return candidate
    for parent in script_path.parents:
        if (parent / "dataset").exists():
            return parent
    return Path.cwd()


def iter_s1_with_candidates(
    s1_path: Path, s2_index, s3_index, limit: Optional[int]
) -> Iterator[Tuple[str, Dict, Set[str], Set[str]]]:
    """
    Streams train/test_source1.tsv ONCE, yielding
    (entity_id, s1_normalized_record, s2_candidate_ids, s3_candidate_ids).

    This mirrors candidate_generation.iter_candidates's per-row logic
    (normalize -> BlockIndex.get_candidates) but ALSO returns the S1
    normalized record itself, so this script no longer needs a second pass
    over source1 just to re-derive it. It's kept as a small, local,
    Phase-3-only helper rather than changing iter_candidates's signature,
    because evaluate_blocking.py (Phase 2) depends on that exact signature
    and must not be touched.
    """
    seen = 0
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            if limit is not None and seen >= limit:
                return
            row_d = row._asdict()
            entity_id = str(row_d.get("entity_id"))
            norm_name = normalize_business_name(row_d.get("business_name"))
            norm_addr = normalize_address(row_d.get("business_address"))
            norm_country = normalize_country(row_d.get("country"))
            s1_record = {"name": norm_name, "address": norm_addr, "country": norm_country}

            s2_cands: Set[str] = set()
            if s2_index is not None:
                s2_cands, _ = s2_index.get_candidates(norm_name, norm_addr, norm_country)
            s3_cands: Set[str] = set()
            if s3_index is not None:
                s3_cands, _ = s3_index.get_candidates(norm_name, norm_addr, norm_country)

            yield entity_id, s1_record, s2_cands, s3_cands
            seen += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 feature-dataset builder")
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help="Number of S1 entities processed per batch before their candidate "
                              "records are loaded, featurized, written, and released. Smaller = "
                              "lower peak memory but more S2/S3 file re-scans (more I/O time).")
    args = parser.parse_args()

    project_root = find_project_root()
    dataset_dir = project_root / "dataset" / args.dataset
    reports_dir = project_root / "reports"
    features_dir = project_root / "features"
    reports_dir.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)

    s1_path = dataset_dir / f"{args.dataset}_source1.tsv"
    s2_path = dataset_dir / f"{args.dataset}_source2.tsv"
    s3_path = dataset_dir / f"{args.dataset}_source3.tsv"
    gt_path = project_root / "dataset" / "train" / "train_ground_truth.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not p.exists():
            print(f"[build_feature_dataset] ERROR: file not found: {p}")
            sys.exit(1)

    limit = None if args.full else args.sample_size
    print(f"[build_feature_dataset] dataset={args.dataset} sample_size={'FULL' if limit is None else limit} "
          f"batch_size={args.batch_size}")

    t0 = time.time()
    print(f"[build_feature_dataset] Building S2 index from {s2_path.name} ...")
    s2_index = build_index_from_source(s2_path, "S2")
    print(f"  -> {len(s2_index.all_ids):,} S2 records indexed in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[build_feature_dataset] Building S3 index from {s3_path.name} ...")
    s3_index = build_index_from_source(s3_path, "S3")
    print(f"  -> {len(s3_index.all_ids):,} S3 records indexed in {time.time() - t1:.1f}s")

    has_ground_truth = args.dataset == "train" and gt_path.exists()
    ground_truth: Dict[str, Dict[str, Set[str]]] = {}
    if has_ground_truth:
        print(f"[build_feature_dataset] Loading ground truth from {gt_path.name} ...")
        ground_truth = load_ground_truth(gt_path, s2_index.all_ids, s3_index.all_ids)
        print(f"  -> {len(ground_truth):,} S1 entities with ground truth loaded")
    else:
        print("[build_feature_dataset] No ground truth for this run — output will have label=None "
              "for every row (test dataset has no train_ground_truth.tsv equivalent).")

    out_path = features_dir / f"{args.dataset}_pair_features.csv"
    print(f"[build_feature_dataset] Streaming S1 in batches of {args.batch_size:,} "
          f"(bounded-memory fix — see module docstring) ...")

    duplicate_count = 0
    n_rows = 0
    n_positive = 0
    n_negative = 0
    n_entities_processed = 0
    missing_value_counts = {name: 0 for name in FEATURE_NAMES}
    unresolvable_candidates = 0  # candidate id present in blocking output but missing from record lookup
    max_batch_needed_ids = 0     # for the report: confirms peak stayed bounded, not global

    s1_gen = iter_s1_with_candidates(s1_path, s2_index, s3_index, limit)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        while True:
            batch: List[Tuple[str, Dict, Set[str], Set[str]]] = list(itertools.islice(s1_gen, args.batch_size))
            if not batch:
                break
            n_entities_processed += len(batch)

            # Needed ids scoped to THIS batch only -- a small fraction of
            # the global total, unlike the previous implementation.
            needed_s2_ids: Set[str] = set()
            needed_s3_ids: Set[str] = set()
            for _, _, s2_cands, s3_cands in batch:
                needed_s2_ids.update(s2_cands)
                needed_s3_ids.update(s3_cands)
            max_batch_needed_ids = max(max_batch_needed_ids, len(needed_s2_ids) + len(needed_s3_ids))

            s2_records = build_record_lookup_for_ids(s2_path, needed_s2_ids)
            s3_records = build_record_lookup_for_ids(s3_path, needed_s3_ids)

            # Duplicate-pair guard is scoped per batch, not globally: each
            # S1 entity is visited exactly once across the whole run (single
            # linear scan of source1), so no two batches can ever produce
            # the same (s1_id, source, candidate_id) triple -- a global set
            # would only grow without ever catching a cross-batch duplicate.
            seen_pairs_in_batch: Set[tuple] = set()

            for s1_id, s1_rec, s2_cands, s3_cands in batch:
                gt = ground_truth.get(s1_id, {"s2": set(), "s3": set()}) if has_ground_truth else None

                for source, cand_ids, records in (("S2", s2_cands, s2_records), ("S3", s3_cands, s3_records)):
                    for cand_id in cand_ids:
                        pair_key = (s1_id, source, cand_id)
                        if pair_key in seen_pairs_in_batch:
                            duplicate_count += 1
                            continue
                        seen_pairs_in_batch.add(pair_key)

                        cand_rec = records.get(cand_id)
                        if cand_rec is None:
                            unresolvable_candidates += 1
                            continue

                        feats = extract_pair_features(s1_rec, cand_rec)
                        for fname, fval in feats.items():
                            if fval == -1:
                                missing_value_counts[fname] += 1

                        if gt is not None:
                            label = 1 if cand_id in gt[source.lower()] else 0
                            n_positive += label
                            n_negative += (1 - label)
                        else:
                            label = ""

                        writer.writerow([s1_id, cand_id, source] + [feats[n] for n in FEATURE_NAMES] + [label])
                        n_rows += 1

            print(f"  ... {n_entities_processed:,} S1 entities processed, {n_rows:,} pair rows written "
                  f"so far (this batch needed {len(needed_s2_ids):,} S2 + {len(needed_s3_ids):,} S3 ids)")

            # Explicitly release this batch's structures before the next
            # iteration allocates a new batch's — the core of the memory fix.
            del needed_s2_ids, needed_s3_ids, s2_records, s3_records, seen_pairs_in_batch, batch

    elapsed = time.time() - t0
    print(f"  -> wrote {n_rows:,} pair rows to {out_path} in {elapsed:.1f}s")

    # --- data-quality / Phase-3 report ---
    class_balance = None
    if has_ground_truth:
        total_labeled = n_positive + n_negative
        class_balance = {
            "positive": n_positive,
            "negative": n_negative,
            "positive_rate": round(n_positive / total_labeled, 4) if total_labeled else None,
        }

    report_lines = [
        "# Phase 3 Feature Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- Dataset: `{args.dataset}`",
        f"- S1 entities processed: {n_entities_processed:,} ({'full' if limit is None else f'first {limit}'})",
        f"- Batch size used: {args.batch_size:,} (memory-fix control; see module docstring)",
        f"- Largest single batch's combined needed-ID count: {max_batch_needed_ids:,} "
        f"(peak memory is bounded by this, not by the run's total distinct candidate ids)",
        f"- Candidate pair rows written: {n_rows:,}",
        f"- Duplicate candidate pairs skipped (already seen): {duplicate_count:,}",
        f"- Candidate ids present in blocking output but unresolvable in source file "
        f"(data-quality flag, should be 0): {unresolvable_candidates:,}",
        f"- Output file: `{out_path.relative_to(project_root)}`\n",
    ]

    if class_balance:
        report_lines += [
            "## Label / class balance\n",
            f"- Positive pairs (label=1): {class_balance['positive']:,}",
            f"- Negative pairs (label=0): {class_balance['negative']:,}",
            f"- Positive rate: {class_balance['positive_rate']}",
            "- Class imbalance is expected and typical for candidate-pair datasets in entity "
            "resolution (most blocked candidates are non-matches); this is a Phase 4 modeling "
            "consideration (e.g. class weighting or resampling), not a Phase 3 bug.\n",
        ]
    else:
        report_lines += ["## Label / class balance\n", "_No ground truth for this run (test dataset)._\n"]

    report_lines += ["## Missing-value counts per feature (sentinel = -1, meaning 'no data', not 'non-match')\n"]
    for fname, cnt in missing_value_counts.items():
        pct = round(cnt / n_rows * 100, 2) if n_rows else 0
        report_lines.append(f"- `{fname}`: {cnt:,} ({pct}%)")

    report_lines += [
        "\n## Label-leakage check\n",
        "- Features are computed exclusively from `normalize_business_name` / `normalize_address` / "
        "`normalize_country` outputs on the S1 record and the candidate record. Ground truth is used "
        "ONLY to assign the `label` column after features are already fixed, never as a feature input, "
        "so no ground-truth information leaks into the feature vector itself.",
        "- `source1_entity_id` / `candidate_entity_id` are kept in the output for traceability but are "
        "raw dataset IDs, not derived from any label — they carry no leakage risk on their own, but "
        "should not be used as model features directly (arbitrary IDs have no generalizable signal).\n",
        "## Known limitation (inherent to the two-stage blocking + classification design)\n",
        "- This dataset can only contain a positive pair if blocking actually produced that candidate. "
        "Phase 2 measured blocking recall on this same run's entities; any true match blocking missed "
        "is absent here as a positive row (it simply never appears as a candidate), not mislabeled as "
        "negative. Phase 4's achievable recall is therefore capped by Phase 2's blocking recall — "
        "improving it later means revisiting blocking, not the classifier.\n",
        "## Memory-fix note (this run)\n",
        f"- Processed in batches of {args.batch_size:,} S1 entities. The largest single batch needed "
        f"{max_batch_needed_ids:,} candidate ids resolved at once, versus the run's full distinct-id "
        f"total which would be much larger if computed globally — this bound is what avoids the "
        f"previous out-of-memory failure. If this run still uses too much memory, re-run with a "
        f"smaller `--batch-size`; if it finishes with room to spare, a larger `--batch-size` will "
        f"finish faster (fewer S2/S3 re-scans) at the cost of higher peak memory.\n",
    ]

    report_path = reports_dir / "phase3_feature_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 70)
    print("PHASE 3 FEATURE DATASET COMPLETE")
    print("=" * 70)
    print(f"Rows: {n_rows:,}  Duplicates skipped: {duplicate_count:,}  "
          f"Unresolvable candidates: {unresolvable_candidates:,}")
    print(f"Largest single batch needed-id count: {max_batch_needed_ids:,} (batch_size={args.batch_size:,})")
    if class_balance:
        print(f"Positive: {class_balance['positive']:,}  Negative: {class_balance['negative']:,}  "
              f"Positive rate: {class_balance['positive_rate']}")
    print(f"Dataset: {out_path}")
    print(f"Report: {report_path}")
    print("STOP — Phase 3 only. No model was trained.")


if __name__ == "__main__":
    main()
