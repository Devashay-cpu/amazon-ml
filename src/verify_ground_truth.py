"""
Ground-truth parsing verification (Phase 1 bug investigation).

The original profiler reported zero=2,206,821 / one=0 / multi=0, which is
almost certainly WRONG. Root cause: profile_ground_truth() looked for
separate S2-match / S3-match columns (candidates containing "s2"/"s3"),
but the real schema uses ONE combined column: `matched_entity_ids`
(comma-separated). Since no column matched the old candidate lists,
s2_col and s3_col were both None, so every row's match count was
computed as 0 -- a silent column-detection failure, not a real result.

This script:
  1. Prints the raw header + first 20-50 raw lines (repr'd) with no pandas
     involved, so we see exact bytes/delimiters/whitespace.
  2. Loads the file with pandas (dtype=str) and shows exact column names.
  3. Shows repr() of several matched_entity_ids values (empty + non-empty).
  4. Sanity-checks the TSV delimiter (tab count per line == header field count).
  5. Re-splits matched_entity_ids correctly and recomputes:
       - empty vs non-empty counts
       - ids-per-row distribution
       - zero/one/multi-match counts, min/max/mean
  6. If files with "source2"/"source3" in their name are found under
     dataset/, builds ID sets from just their ID column (not the whole
     file) to split each row's matched IDs into S2-matches vs S3-matches,
     and counts rows with both.

It does NOT re-profile the multi-gigabyte source files, and does NOT
touch normalization / blocking / modeling.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

CHUNK_SIZE = 100_000
GROUND_TRUTH_FILENAME_HINTS = ["ground_truth", "groundtruth", "train_ground_truth"]

S1_ID_CANDIDATES = ["source1_entity_id", "s1_id", "source1_id", "s1", "id", "entity_id"]
MATCH_COLUMN_CANDIDATES = [
    "matched_entity_ids", "matched_ids", "matches", "match_ids", "matching_ids",
]

LIST_SPLIT_CHARS = [";", "|", ","]


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent
    if (candidate / "dataset").exists():
        return candidate
    for parent in script_path.parents:
        if (parent / "dataset").exists():
            return parent
    return Path.cwd()


def find_ground_truth_file(dataset_dir: Path) -> Optional[Path]:
    for split in ("train", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob("*.tsv"):
            if any(hint in path.stem.lower() for hint in GROUND_TRUTH_FILENAME_HINTS):
                return path
    return None


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        for col_lower, col_original in lower_map.items():
            if cand.lower() in col_lower:
                return col_original
    return None


def split_id_list(cell) -> List[str]:
    if cell is None:
        return []
    text = str(cell).strip()
    if text == "" or text.lower() in ("nan", "none", "null", "-", "[]"):
        return []
    text = text.strip("[]() ")
    for ch in LIST_SPLIT_CHARS:
        if ch in text:
            text = text.replace(ch, "|")
    parts = [p.strip().strip("'\"") for p in text.split("|")]
    return [p for p in parts if p not in ("", "nan", "none", "null")]


def find_source_id_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    """Locate files whose name suggests Source 2 / Source 3 records."""
    found: Dict[str, List[Path]] = {"source2": [], "source3": []}
    for path in dataset_dir.rglob("*.tsv"):
        stem = path.stem.lower()
        if "source2" in stem or stem.endswith("_s2") or "_s2_" in stem:
            found["source2"].append(path)
        elif "source3" in stem or stem.endswith("_s3") or "_s3_" in stem:
            found["source3"].append(path)
    return found


def load_id_set(path: Path) -> Set[str]:
    """Load only the ID column of a file into a set (not the full file)."""
    header_cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    id_col = find_column(header_cols, ["id", "entity_id", "record_id", "poi_id", "index"])
    if id_col is None:
        return set()
    ids: Set[str] = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=[id_col], dtype=str, chunksize=CHUNK_SIZE):
        ids.update(chunk[id_col].dropna().astype(str).tolist())
    return ids


def main() -> None:
    project_root = find_project_root()
    dataset_dir = project_root / "dataset"
    gt_path = find_ground_truth_file(dataset_dir)

    if gt_path is None:
        log("ERROR: could not find a train_ground_truth.tsv-style file under dataset/train.")
        sys.exit(1)

    log(f"Ground-truth file: {gt_path}")

    # ---- 1. Raw peek, no pandas ----
    log("=" * 70)
    log("STEP 1-3: RAW FILE INSPECTION (no pandas)")
    log("=" * 70)
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [f.readline().rstrip("\n").rstrip("\r") for _ in range(51)]

    header_line = raw_lines[0]
    header_cols = header_line.split("\t")
    log(f"Raw header line: {header_line!r}")
    log(f"Header field count (split on real tab): {len(header_cols)}")
    log(f"Columns: {header_cols}")

    mismatches = 0
    for i, line in enumerate(raw_lines[1:21], start=1):
        n_tabs_fields = len(line.split("\t"))
        if n_tabs_fields != len(header_cols):
            mismatches += 1
        log(f"  row {i}: fields={n_tabs_fields} raw={line!r}")
    log(f"Tab-delimiter sanity check: {mismatches} of first 20 data rows have a different "
        f"field count than the header ({'OK' if mismatches == 0 else 'INVESTIGATE'}).")

    # ---- 2. pandas load, exact column names ----
    log("=" * 70)
    log("STEP 2: PANDAS COLUMN NAMES")
    log("=" * 70)
    df_head = pd.read_csv(gt_path, sep="\t", dtype=str, nrows=5)
    pandas_cols = df_head.columns.tolist()
    log(f"pandas columns: {pandas_cols}")

    s1_col = find_column(pandas_cols, S1_ID_CANDIDATES)
    match_col = find_column(pandas_cols, MATCH_COLUMN_CANDIDATES)
    log(f"Detected S1 id column: {s1_col!r}")
    log(f"Detected combined match column: {match_col!r}")

    if match_col is None:
        log("ERROR: still could not detect the match column even with the corrected "
            "candidate list. Print `pandas_cols` above and add the exact header name "
            "to MATCH_COLUMN_CANDIDATES in this script and in data_profiler.py.")
        sys.exit(1)

    # ---- 3. repr() of sample values ----
    log("=" * 70)
    log("STEP 3: repr() OF matched_entity_ids VALUES (first 20 rows)")
    log("=" * 70)
    for i, val in enumerate(df_head[match_col].tolist()):
        log(f"  row {i}: repr={val!r}  type={type(val).__name__}  is_na={pd.isna(val)}")

    # ---- 4/5/6: full streaming recount ----
    log("=" * 70)
    log("STEP 4-6: FULL STREAMING RE-COUNT OF matched_entity_ids")
    log("=" * 70)

    total_rows = 0
    empty_count = 0
    non_empty_count = 0
    ids_per_row_counter: Counter = Counter()
    non_empty_examples: List[Dict] = []

    reader = pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE)
    for chunk in reader:
        total_rows += len(chunk)
        for _, row in chunk.iterrows():
            raw_val = row.get(match_col)
            ids = split_id_list(raw_val)
            n = len(ids)
            ids_per_row_counter[n] += 1
            if n == 0:
                empty_count += 1
            else:
                non_empty_count += 1
                if len(non_empty_examples) < 15:
                    non_empty_examples.append({
                        "s1_id": row.get(s1_col) if s1_col else None,
                        "raw": raw_val,
                        "parsed_ids": ids,
                        "n_ids": n,
                    })

    zero_match = ids_per_row_counter.get(0, 0)
    one_match = ids_per_row_counter.get(1, 0)
    multi_match = sum(c for n, c in ids_per_row_counter.items() if n >= 2)
    all_counts = [n for n in ids_per_row_counter.elements()]  # expands counter -> list of per-row n
    min_matches = min(ids_per_row_counter) if ids_per_row_counter else 0
    max_matches = max(ids_per_row_counter) if ids_per_row_counter else 0
    mean_matches = (sum(n * c for n, c in ids_per_row_counter.items()) / total_rows) if total_rows else 0.0

    log(f"Total rows: {total_rows:,}")
    log(f"Empty matched_entity_ids: {empty_count:,}")
    log(f"Non-empty matched_entity_ids: {non_empty_count:,}")
    log(f"Zero-match rows: {zero_match:,}")
    log(f"One-match rows: {one_match:,}")
    log(f"Multi-match rows: {multi_match:,}")
    log(f"Min / Max / Mean matches per row: {min_matches} / {max_matches} / {mean_matches:.4f}")
    log(f"Ids-per-row distribution (first 20 keys): "
        f"{dict(sorted(ids_per_row_counter.items())[:20])}")

    log("Examples with non-empty matched_entity_ids:")
    for ex in non_empty_examples:
        log(f"  s1_id={ex['s1_id']!r} raw={ex['raw']!r} parsed={ex['parsed_ids']} n={ex['n_ids']}")

    # ---- 7. S2 vs S3 split, if source files can be located ----
    log("=" * 70)
    log("STEP 7: S2 vs S3 SPLIT (via ID-set lookup against source files)")
    log("=" * 70)
    source_files = find_source_id_files(dataset_dir)
    log(f"Source2-like files found: {[str(p) for p in source_files['source2']]}")
    log(f"Source3-like files found: {[str(p) for p in source_files['source3']]}")

    if source_files["source2"] or source_files["source3"]:
        s2_ids: Set[str] = set()
        for p in source_files["source2"]:
            log(f"Loading ID column from {p.name} ...")
            s2_ids |= load_id_set(p)
        s3_ids: Set[str] = set()
        for p in source_files["source3"]:
            log(f"Loading ID column from {p.name} ...")
            s3_ids |= load_id_set(p)
        log(f"Loaded {len(s2_ids):,} distinct Source-2 IDs, {len(s3_ids):,} distinct Source-3 IDs.")

        s2_match_counts: Counter = Counter()
        s3_match_counts: Counter = Counter()
        both_s2_s3 = 0
        only_s2 = 0
        only_s3 = 0
        unmatched_ids_sample: List[str] = []

        reader = pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE)
        for chunk in reader:
            for _, row in chunk.iterrows():
                ids = split_id_list(row.get(match_col))
                n_s2 = sum(1 for i in ids if i in s2_ids)
                n_s3 = sum(1 for i in ids if i in s3_ids)
                n_unclassified = len(ids) - n_s2 - n_s3
                if n_unclassified > 0 and len(unmatched_ids_sample) < 10:
                    unmatched_ids_sample.extend(
                        [i for i in ids if i not in s2_ids and i not in s3_ids][:2]
                    )
                s2_match_counts[n_s2] += 1
                s3_match_counts[n_s3] += 1
                if n_s2 > 0 and n_s3 > 0:
                    both_s2_s3 += 1
                elif n_s2 > 0:
                    only_s2 += 1
                elif n_s3 > 0:
                    only_s3 += 1

        log(f"S2 match-count distribution: {dict(sorted(s2_match_counts.items()))}")
        log(f"S3 match-count distribution: {dict(sorted(s3_match_counts.items()))}")
        log(f"Entities with only S2 matches: {only_s2:,}")
        log(f"Entities with only S3 matches: {only_s3:,}")
        log(f"Entities with BOTH S2 and S3 matches: {both_s2_s3:,}")
        if unmatched_ids_sample:
            log(f"WARNING: some matched IDs were not found in either source's ID set, e.g.: "
                f"{unmatched_ids_sample[:10]}")
    else:
        log("Could not locate files with 'source2'/'source3' in their name under dataset/. "
            "Skipping the S2/S3 split -- report total match counts only (already computed above). "
            "If Source 2 / Source 3 files use a different naming convention, tell me the exact "
            "filenames and I will point this script at them.")

    log("=" * 70)
    log("DONE. This script only reads train_ground_truth.tsv (+ ID columns of source2/source3 "
        "files, if found). No other multi-gigabyte files were re-profiled.")


if __name__ == "__main__":
    main()
