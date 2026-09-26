"""
Phase 2 — Blocking evaluation

Usage:
    python src/evaluate_blocking.py                     # small validation run (train, sample)
    python src/evaluate_blocking.py --sample-size 100000 # bigger sample
    python src/evaluate_blocking.py --full               # entire train_source1.tsv
    python src/evaluate_blocking.py --dataset test        # candidate generation only (no recall; test has no ground truth)

Steps (per spec):
  1. Build inverted indexes from S2/S3.
  2. Generate candidates for a manageable subset first (default sample).
  3. Compare candidate pairs against ground truth.
  4. Calculate blocking recall (overall, S2, S3, only-S2, only-S3, both).
  5. Report candidate-count statistics (mean/median/p90/p95/p99/max).

Writes reports/blocking_report.md + reports/blocking_stats.json.
Does NOT implement any ML classifier, scoring, or thresholding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

from candidate_generation import build_index_from_source, iter_candidates
from gt_loader import load_ground_truth


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent
    if (candidate / "dataset").exists():
        return candidate
    for parent in script_path.parents:
        if (parent / "dataset").exists():
            return parent
    return Path.cwd()


def percentile(sorted_values: List[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


def candidate_count_stats(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {}
    s = sorted(counts)
    return {
        "mean": round(mean(counts), 2),
        "median": median(counts),
        "p90": percentile(s, 0.90),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
        "max": s[-1],
        "min": s[0],
    }


def safe_div(numer: int, denom: int) -> Optional[float]:
    """Returns None (not 0.0) when there's no ground truth to measure against,
    so 'no matches found' and 'no data for this group' are never conflated."""
    return round(numer / denom, 4) if denom else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 blocking evaluation")
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--sample-size", type=int, default=20_000,
                         help="Number of Source-1 rows to evaluate (ignored with --full)")
    parser.add_argument("--full", action="store_true", help="Process the entire Source-1 file")
    args = parser.parse_args()

    project_root = find_project_root()
    dataset_dir = project_root / "dataset" / args.dataset
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    s1_path = dataset_dir / f"{args.dataset}_source1.tsv"
    s2_path = dataset_dir / f"{args.dataset}_source2.tsv"
    s3_path = dataset_dir / f"{args.dataset}_source3.tsv"
    gt_path = project_root / "dataset" / "train" / "train_ground_truth.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not p.exists():
            print(f"[evaluate_blocking] ERROR: expected file not found: {p}")
            sys.exit(1)

    limit = None if args.full else args.sample_size
    print(f"[evaluate_blocking] dataset={args.dataset} sample_size={'FULL' if limit is None else limit}")

    tracemalloc.start()
    t0 = time.time()

    print(f"[evaluate_blocking] Building S2 index from {s2_path.name} ...")
    s2_index = build_index_from_source(s2_path, "S2")
    t_s2 = time.time()
    print(f"  -> {len(s2_index.all_ids):,} S2 records indexed in {t_s2 - t0:.1f}s")

    print(f"[evaluate_blocking] Building S3 index from {s3_path.name} ...")
    s3_index = build_index_from_source(s3_path, "S3")
    t_s3 = time.time()
    print(f"  -> {len(s3_index.all_ids):,} S3 records indexed in {t_s3 - t_s2:.1f}s")

    has_ground_truth = args.dataset == "train" and gt_path.exists()
    ground_truth: Dict[str, Dict[str, set]] = {}
    if has_ground_truth:
        print(f"[evaluate_blocking] Loading ground truth from {gt_path.name} ...")
        ground_truth = load_ground_truth(gt_path, s2_index.all_ids, s3_index.all_ids)
        print(f"  -> {len(ground_truth):,} S1 entities with ground truth loaded")
    else:
        print("[evaluate_blocking] No ground truth available for this run "
              "(dataset=test, or train_ground_truth.tsv missing) — recall will not be computed.")

    # --- accumulators ---
    total_entities = 0
    candidate_counts: List[int] = []
    s2_candidate_counts: List[int] = []
    s3_candidate_counts: List[int] = []

    gt_pairs_s2 = 0
    found_pairs_s2 = 0
    gt_pairs_s3 = 0
    found_pairs_s3 = 0

    # group = only_s2 / only_s3 / both / neither, based on this entity's OWN
    # ground truth (not the global Phase-1 aggregate), so the breakdown is
    # exact for whatever subset was actually evaluated.
    group_gt_pairs = {"only_s2": 0, "only_s3": 0, "both": 0}
    group_found_pairs = {"only_s2": 0, "only_s3": 0, "both": 0}

    strategy_pairs_found: Dict[str, int] = {}

    print(f"[evaluate_blocking] Generating candidates for Source-1 rows ...")
    t_cand_start = time.time()

    for entity_id, s2_cands, s3_cands, s2_per_strat, s3_per_strat in iter_candidates(
        s1_path, s2_index, s3_index, limit=limit
    ):
        total_entities += 1
        n_s2 = len(s2_cands)
        n_s3 = len(s3_cands)
        candidate_counts.append(n_s2 + n_s3)
        s2_candidate_counts.append(n_s2)
        s3_candidate_counts.append(n_s3)

        if has_ground_truth:
            gt = ground_truth.get(entity_id, {"s2": set(), "s3": set()})
            truth_s2, truth_s3 = gt["s2"], gt["s3"]

            found_s2 = truth_s2 & s2_cands
            found_s3 = truth_s3 & s3_cands

            gt_pairs_s2 += len(truth_s2)
            found_pairs_s2 += len(found_s2)
            gt_pairs_s3 += len(truth_s3)
            found_pairs_s3 += len(found_s3)

            if truth_s2 and truth_s3:
                group = "both"
            elif truth_s2:
                group = "only_s2"
            elif truth_s3:
                group = "only_s3"
            else:
                group = None
            if group:
                group_gt_pairs[group] += len(truth_s2) + len(truth_s3)
                group_found_pairs[group] += len(found_s2) + len(found_s3)

            for strat, ids in s2_per_strat.items():
                strategy_pairs_found[strat] = strategy_pairs_found.get(strat, 0) + len(truth_s2 & ids)
            for strat, ids in s3_per_strat.items():
                strategy_pairs_found[strat] = strategy_pairs_found.get(strat, 0) + len(truth_s3 & ids)

        if total_entities % 5000 == 0:
            print(f"  ... {total_entities:,} entities processed")

    t_cand_end = time.time()
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    overall_gt_pairs = gt_pairs_s2 + gt_pairs_s3
    overall_found_pairs = found_pairs_s2 + found_pairs_s3

    stats = {
        "run": {
            "dataset": args.dataset,
            "entities_evaluated": total_entities,
            "sample_mode": "full" if limit is None else f"first {limit}",
            "s2_index_build_seconds": round(t_s2 - t0, 2),
            "s3_index_build_seconds": round(t_s3 - t_s2, 2),
            "candidate_generation_seconds": round(t_cand_end - t_cand_start, 2),
            "peak_memory_mb": round(peak_mem, 1),
        },
        "candidate_count_stats": {
            "total": candidate_count_stats(candidate_counts),
            "s2_only": candidate_count_stats(s2_candidate_counts),
            "s3_only": candidate_count_stats(s3_candidate_counts),
        },
        "blocking_recall": {
            "overall": safe_div(overall_found_pairs, overall_gt_pairs),
            "s2": safe_div(found_pairs_s2, gt_pairs_s2),
            "s3": safe_div(found_pairs_s3, gt_pairs_s3),
            "only_s2_entities": safe_div(group_found_pairs["only_s2"], group_gt_pairs["only_s2"]),
            "only_s3_entities": safe_div(group_found_pairs["only_s3"], group_gt_pairs["only_s3"]),
            "both_s2_s3_entities": safe_div(group_found_pairs["both"], group_gt_pairs["both"]),
        } if has_ground_truth else None,
        "raw_pair_counts": {
            "gt_pairs_s2": gt_pairs_s2, "found_pairs_s2": found_pairs_s2,
            "gt_pairs_s3": gt_pairs_s3, "found_pairs_s3": found_pairs_s3,
            "group_gt_pairs": group_gt_pairs, "group_found_pairs": group_found_pairs,
        } if has_ground_truth else None,
        "strategy_pairs_found": strategy_pairs_found if has_ground_truth else None,
        "s2_problematic_blocks": s2_index.problematic_blocks(),
        "s3_problematic_blocks": s3_index.problematic_blocks(),
        "s2_strategy_key_counts": s2_index.strategy_key_counts(),
        "s3_strategy_key_counts": s3_index.strategy_key_counts(),
    }

    json_path = reports_dir / "blocking_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    md_path = reports_dir / "blocking_report.md"
    md_path.write_text(build_markdown_report(stats), encoding="utf-8")

    print("=" * 70)
    print("BLOCKING EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Entities evaluated: {total_entities:,}")
    def fmt_recall(v):
        return "N/A (no ground-truth pairs in this group)" if v is None else f"{v:.2%}"

    if has_ground_truth:
        r = stats["blocking_recall"]
        print(f"Overall blocking recall: {fmt_recall(r['overall'])}")
        print(f"  S2 recall: {fmt_recall(r['s2'])}   S3 recall: {fmt_recall(r['s3'])}")
        print(f"  only-S2 recall: {fmt_recall(r['only_s2_entities'])}   "
              f"only-S3 recall: {fmt_recall(r['only_s3_entities'])}   "
              f"both recall: {fmt_recall(r['both_s2_s3_entities'])}")
    print(f"Candidate count (total) stats: {stats['candidate_count_stats']['total']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print("STOP — Phase 2 blocking evaluation only. Review before scaling to the full dataset "
          "or moving to Phase 3 (feature engineering / scoring / ML).")


def build_markdown_report(stats: Dict) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Blocking Report (Phase 2)\n")
    run = stats["run"]
    a(f"- Dataset: `{run['dataset']}`")
    a(f"- Entities evaluated: {run['entities_evaluated']:,} ({run['sample_mode']})")
    a(f"- S2 index build time: {run['s2_index_build_seconds']}s")
    a(f"- S3 index build time: {run['s3_index_build_seconds']}s")
    a(f"- Candidate generation time: {run['candidate_generation_seconds']}s")
    a(f"- Peak traced memory: {run['peak_memory_mb']} MB\n")

    a("## Strategies tested\n")
    for s in ["country_exact_name", "country_name_prefix4", "country_suffix_stripped_name",
              "country_rare_name_token", "postal_code", "house_number_street",
              "address_alnum_prefix8"]:
        a(f"- `{s}`")
    a("")

    def fmt_recall(v):
        return "N/A (no ground-truth pairs in this group)" if v is None else f"{v:.2%}"

    if stats["blocking_recall"]:
        r = stats["blocking_recall"]
        a("## Blocking recall\n")
        a(f"- Overall: **{fmt_recall(r['overall'])}**")
        a(f"- S2: {fmt_recall(r['s2'])}")
        a(f"- S3: {fmt_recall(r['s3'])}")
        a(f"- Entities with only S2 matches: {fmt_recall(r['only_s2_entities'])}")
        a(f"- Entities with only S3 matches: {fmt_recall(r['only_s3_entities'])}")
        a(f"- Entities with both S2 and S3 matches: {fmt_recall(r['both_s2_s3_entities'])}\n")

        a("## Strategy contribution (true pairs recovered by each strategy)\n")
        a("Note: a pair can be recovered by more than one strategy; these are not additive.\n")
        for strat, n in sorted(stats["strategy_pairs_found"].items(), key=lambda kv: -kv[1]):
            a(f"- `{strat}`: {n:,} true pairs recovered")
        a("")
    else:
        a("## Blocking recall\n")
        a("_No ground truth available for this run (test dataset, or file missing)._\n")

    a("## Candidate-count statistics (per Source-1 entity)\n")
    for label, key in [("Total (S2+S3)", "total"), ("S2 only", "s2_only"), ("S3 only", "s3_only")]:
        s = stats["candidate_count_stats"][key]
        if s:
            a(f"**{label}** — mean={s['mean']}, median={s['median']}, p90={s['p90']}, "
              f"p95={s['p95']}, p99={s['p99']}, max={s['max']}, min={s['min']}")
    a("")

    a("## Problematic (oversized) blocks\n")
    for label, key in [("S2", "s2_problematic_blocks"), ("S3", "s3_problematic_blocks")]:
        blocks = stats[key]
        a(f"**{label}** — {len(blocks)} block(s) over threshold:")
        for strategy, block_key, size in blocks[:10]:
            a(f"- `{strategy}` key=`{block_key}` size={size:,}")
        a("")

    a("## Strategy key-space size (number of distinct blocking keys generated)\n")
    for label, key in [("S2", "s2_strategy_key_counts"), ("S3", "s3_strategy_key_counts")]:
        a(f"**{label}**")
        for strat, n in stats[key].items():
            a(f"- `{strat}`: {n:,} distinct keys")
    a("")

    a("## Memory / runtime observations\n")
    a(f"- Peak traced Python memory during this run: {stats['run']['peak_memory_mb']} MB.")
    a("- Both source files are read twice when building each index (pass 1: name-token "
      "frequency only; pass 2: full index) to avoid holding every normalized record in "
      "memory between passes — a deliberate CPU-for-memory tradeoff.")
    a("- No all-pairs comparison is performed anywhere; every candidate lookup is a dict "
      "hit against a precomputed inverted index.")
    a("- For the full ~2.2M-row training set, in-memory dict-of-lists indexes are expected "
      "to scale to tens of millions of keys; if memory becomes a constraint, the recommended "
      "next step is a disk-backed index (e.g. SQLite with an indexed (strategy, key) -> id "
      "table) rather than changing the blocking logic itself.\n")

    a("## Recommended final blocking strategy\n")
    if stats["blocking_recall"]:
        top_strats = sorted(stats["strategy_pairs_found"].items(), key=lambda kv: -kv[1])[:3]
        names = ", ".join(f"`{s}`" for s, _ in top_strats)
        a(f"Based on this run, the strategies recovering the most true pairs were: {names}. "
          "Recommend keeping the full strategy set for Phase 3 candidate generation (recall "
          "is the priority at this stage), while monitoring the 'problematic blocks' above as "
          "targets for future refinement (e.g. combining a rare-token key with a secondary "
          "signal) once precision-side tuning begins.")
    else:
        a("_Run this script with `--dataset train` to get a recall-based recommendation._")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
