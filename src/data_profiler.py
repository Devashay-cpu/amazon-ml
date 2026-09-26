"""
Phase 1 — Dataset Profiler
Amazon ML Challenge: Business Entity Resolution

Memory-efficient, chunked profiling of the challenge dataset.
Does NOT implement normalization, blocking, ML, embeddings, matching,
or test prediction. Profiling / EDA ONLY.

Run from the project root (the folder that contains `dataset/`, `utils/`,
`README.md`, i.e. D:\\amazon-ml\\student_resource):

    python src/data_profiler.py

Outputs:
    reports/dataset_profile.json
    reports/dataset_profile.md
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

CHUNK_SIZE = 100_000          # adjust down if you hit memory pressure
RANDOM_SEED = 42
DETERMINISTIC_HEAD_ROWS = 20
RANDOM_SAMPLE_ROWS = 50
RESERVOIR_SIZE_FOR_LENGTHS = 20_000   # approx-percentile reservoir per column
PER_COUNTRY_EXAMPLES = 3

random.seed(RANDOM_SEED)

# Candidate column names — the profiler auto-detects whichever of these
# (case-insensitive, substring-tolerant) actually exist in each file, so it
# does not assume one fixed schema across Source 1 / Source 2 / Source 3 /
# ground truth files.
ID_COLUMN_CANDIDATES = ["id", "entity_id", "record_id", "poi_id", "index"]
NAME_COLUMN_CANDIDATES = ["business_name", "name", "poi_name", "entity_name"]
ADDRESS_COLUMN_CANDIDATES = ["business_address", "address", "poi_address", "full_address"]
COUNTRY_COLUMN_CANDIDATES = ["country", "country_code", "nation"]

# NOTE (bug fix, see verify_ground_truth.py): the real ground-truth schema uses
# source1_entity_id + a single combined, comma-separated matched_entity_ids
# column -- not separate S2-match / S3-match columns. GT_S2_MATCH_CANDIDATES /
# GT_S3_MATCH_CANDIDATES below are kept only as a fallback for a schema variant
# that does split them; GT_MATCH_COLUMN_CANDIDATES is checked first and is what
# actually matches this dataset.
GT_S1_ID_CANDIDATES = ["source1_entity_id", "s1_id", "source1_id", "s1", "id", "entity_id"]
GT_MATCH_COLUMN_CANDIDATES = ["matched_entity_ids", "matched_ids", "matches", "match_ids", "matching_ids"]
GT_S2_MATCH_CANDIDATES = ["s2_ids", "s2_id", "source2_ids", "source2_matches", "s2_matches", "s2"]
GT_S3_MATCH_CANDIDATES = ["s3_ids", "s3_id", "source3_ids", "source3_matches", "s3_matches", "s3"]

# Filename substrings used to locate Source 2 / Source 3 record files, so that
# matched_entity_ids can be split into S2-matches vs S3-matches by ID-set
# membership when the ground truth doesn't keep them in separate columns.
SOURCE2_FILENAME_HINTS = ["source2"]
SOURCE3_FILENAME_HINTS = ["source3"]

GROUND_TRUTH_FILENAME_HINTS = ["ground_truth", "groundtruth", "train_ground_truth"]

LIST_SPLIT_CHARS = [";", "|", ","]


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[profiler] {msg}", flush=True)


def find_project_root() -> Path:
    """
    Resolve the project root relative to this script's location, so the
    profiler works regardless of the OS or current working directory.
    Expected layout:
        <root>/src/data_profiler.py   (this file)
        <root>/dataset/train
        <root>/dataset/test
    """
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent  # .../student_resource
    if (candidate / "dataset").exists():
        return candidate
    # Fallback: walk up looking for a `dataset` directory.
    for parent in script_path.parents:
        if (parent / "dataset").exists():
            return parent
    # Last resort: current working directory.
    return Path.cwd()


def find_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitive exact match first, then substring match."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        for col_lower, col_original in lower_map.items():
            if cand.lower() in col_lower:
                return col_original
    return None


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def split_id_list(cell: Any) -> List[str]:
    """Split a delimited ID-list cell into individual, cleaned tokens."""
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


def sniff_delimiter_and_columns(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").rstrip("\r")
    return header.split("\t")


# --------------------------------------------------------------------------
# RUNNING STATS (memory-light, streaming)
# --------------------------------------------------------------------------

@dataclass
class RunningLengthStats:
    """Streaming mean/std/min/max plus an approximate-percentile reservoir."""
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    reservoir: List[int] = field(default_factory=list)
    _seen: int = 0

    def update(self, lengths: Iterable[int]) -> None:
        for length in lengths:
            self.count += 1
            self.total += length
            self.total_sq += length * length
            self.minimum = length if self.minimum is None else min(self.minimum, length)
            self.maximum = length if self.maximum is None else max(self.maximum, length)
            self._seen += 1
            if len(self.reservoir) < RESERVOIR_SIZE_FOR_LENGTHS:
                self.reservoir.append(length)
            else:
                j = random.randint(0, self._seen - 1)
                if j < RESERVOIR_SIZE_FOR_LENGTHS:
                    self.reservoir[j] = length

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        std = variance ** 0.5
        pct = {}
        if self.reservoir:
            s = sorted(self.reservoir)
            for p in (25, 50, 75, 90, 99):
                idx = min(int(len(s) * p / 100), len(s) - 1)
                pct[f"p{p}"] = s[idx]
        return {
            "count": self.count,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": self.minimum,
            "max": self.maximum,
            "approx_percentiles": pct,
            "note": "percentiles are approximate, computed from a fixed-seed reservoir sample",
        }


@dataclass
class ColumnFileProfile:
    file_name: str
    file_size_bytes: int
    columns: List[str]
    row_count: int = 0
    missing_counts: Dict[str, int] = field(default_factory=dict)
    unique_id_count: Optional[int] = None
    id_column_used: Optional[str] = None
    country_distribution: Counter = field(default_factory=Counter)
    country_column_used: Optional[str] = None
    name_length_stats: Optional[RunningLengthStats] = None
    name_column_used: Optional[str] = None
    address_length_stats: Optional[RunningLengthStats] = None
    address_column_used: Optional[str] = None
    duration_seconds: float = 0.0


# --------------------------------------------------------------------------
# CORE PROFILING
# --------------------------------------------------------------------------

def discover_tsv_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {"train": [], "test": []}
    for split in ("train", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists():
            result[split] = sorted(split_dir.glob("*.tsv"))
    return result


def is_ground_truth_file(file_path: Path) -> bool:
    stem = file_path.stem.lower()
    return any(hint in stem for hint in GROUND_TRUTH_FILENAME_HINTS)


def profile_entity_file(file_path: Path) -> ColumnFileProfile:
    start = time.time()
    file_size = file_path.stat().st_size
    columns = sniff_delimiter_and_columns(file_path)

    id_col = find_column(columns, ID_COLUMN_CANDIDATES)
    name_col = find_column(columns, NAME_COLUMN_CANDIDATES)
    address_col = find_column(columns, ADDRESS_COLUMN_CANDIDATES)
    country_col = find_column(columns, COUNTRY_COLUMN_CANDIDATES)

    profile = ColumnFileProfile(
        file_name=file_path.name,
        file_size_bytes=file_size,
        columns=columns,
        id_column_used=id_col,
        name_column_used=name_col,
        address_column_used=address_col,
        country_column_used=country_col,
        name_length_stats=RunningLengthStats() if name_col else None,
        address_length_stats=RunningLengthStats() if address_col else None,
    )

    missing_counts: Counter = Counter()
    unique_ids: set = set()
    track_unique_ids = True  # disabled automatically if the ID space is too large to hold in memory safely

    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL"],
    )

    for chunk in reader:
        profile.row_count += len(chunk)

        for col in columns:
            if col in chunk.columns:
                missing_counts[col] += int(chunk[col].isna().sum())

        if id_col and track_unique_ids:
            unique_ids.update(chunk[id_col].dropna().astype(str).tolist())
            if len(unique_ids) > 5_000_000:  # safety valve against runaway memory use
                track_unique_ids = False
                unique_ids.clear()

        if country_col and country_col in chunk.columns:
            profile.country_distribution.update(chunk[country_col].dropna().astype(str).tolist())

        if name_col and name_col in chunk.columns:
            lengths = chunk[name_col].dropna().astype(str).str.len().tolist()
            profile.name_length_stats.update(lengths)

        if address_col and address_col in chunk.columns:
            lengths = chunk[address_col].dropna().astype(str).str.len().tolist()
            profile.address_length_stats.update(lengths)

    profile.missing_counts = dict(missing_counts)
    profile.unique_id_count = len(unique_ids) if track_unique_ids and id_col else None
    profile.duration_seconds = round(time.time() - start, 2)
    return profile


def deterministic_and_random_sample(file_path: Path, columns: List[str]) -> Dict[str, Any]:
    """Small, reproducible samples for qualitative inspection. Never loads the full file."""
    head_rows: List[Dict[str, Any]] = []
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    reader = pd.read_csv(file_path, sep="\t", chunksize=CHUNK_SIZE, dtype=str)
    for chunk in reader:
        records = chunk.to_dict(orient="records")
        for rec in records:
            if len(head_rows) < DETERMINISTIC_HEAD_ROWS:
                head_rows.append(rec)
            seen += 1
            if len(reservoir) < RANDOM_SAMPLE_ROWS:
                reservoir.append(rec)
            else:
                j = random.randint(0, seen - 1)
                if j < RANDOM_SAMPLE_ROWS:
                    reservoir[j] = rec

    return {
        "head_sample": head_rows,
        "random_sample_seed": RANDOM_SEED,
        "random_sample": reservoir,
    }


def per_country_examples(file_path: Path, country_col: Optional[str], name_col: Optional[str],
                          address_col: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """A few example rows per country, collected in a single streaming pass, capped in size."""
    if not country_col:
        return {}
    examples: Dict[str, List[Dict[str, Any]]] = {}
    max_countries_tracked = 300  # safety valve

    reader = pd.read_csv(file_path, sep="\t", chunksize=CHUNK_SIZE, dtype=str)
    for chunk in reader:
        if country_col not in chunk.columns:
            continue
        for _, row in chunk.iterrows():
            country = row.get(country_col)
            if pd.isna(country):
                continue
            country = str(country)
            bucket = examples.setdefault(country, [])
            if len(examples) > max_countries_tracked and country not in examples:
                continue
            if len(bucket) < PER_COUNTRY_EXAMPLES:
                bucket.append({
                    "id": row.get(find_column(chunk.columns, ID_COLUMN_CANDIDATES) or "", None),
                    "name": row.get(name_col) if name_col else None,
                    "address": row.get(address_col) if address_col else None,
                })
    return examples


def find_source_id_files(dataset_dir: Path, hints: List[str]) -> List[Path]:
    """Locate files whose name suggests Source 2 / Source 3 records, anywhere under dataset/."""
    matches = []
    for path in dataset_dir.rglob("*.tsv"):
        stem = path.stem.lower()
        if any(hint in stem for hint in hints):
            matches.append(path)
    return matches


def load_id_set(path: Path) -> set:
    """Load only a file's ID column into a set — not the whole file — to keep memory low."""
    header_cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    id_col = find_column(header_cols, ID_COLUMN_CANDIDATES)
    if id_col is None:
        return set()
    ids: set = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=[id_col], dtype=str, chunksize=CHUNK_SIZE):
        ids.update(chunk[id_col].dropna().astype(str).tolist())
    return ids


def profile_ground_truth(file_path: Path, dataset_dir: Path) -> Dict[str, Any]:
    columns = sniff_delimiter_and_columns(file_path)
    s1_col = find_column(columns, GT_S1_ID_CANDIDATES)

    # Prefer a single combined matched-IDs column (this dataset's actual schema:
    # source1_entity_id + matched_entity_ids). Fall back to separate S2/S3
    # columns only if no combined column is found.
    match_col = find_column(columns, GT_MATCH_COLUMN_CANDIDATES)
    s2_col = None if match_col else find_column(columns, GT_S2_MATCH_CANDIDATES)
    s3_col = None if match_col else find_column(columns, GT_S3_MATCH_CANDIDATES)

    # If we only have a combined column, try to split S2 vs S3 by ID-set
    # membership against any source2/source3 files we can find.
    s2_ids_set = None
    s3_ids_set = None
    if match_col:
        s2_files = find_source_id_files(dataset_dir, SOURCE2_FILENAME_HINTS)
        s3_files = find_source_id_files(dataset_dir, SOURCE3_FILENAME_HINTS)
        if s2_files or s3_files:
            s2_ids_set = set()
            for p in s2_files:
                s2_ids_set |= load_id_set(p)
            s3_ids_set = set()
            for p in s3_files:
                s3_ids_set |= load_id_set(p)

    total_rows = 0
    zero_match = 0
    one_match = 0
    multi_match = 0
    match_count_distribution: Counter = Counter()
    s2_match_counts: Counter = Counter()
    s3_match_counts: Counter = Counter()
    both_s2_and_s3 = 0
    s1_with_only_s2 = 0
    s1_with_only_s3 = 0
    max_matches_seen = 0
    max_matches_example_id = None
    unclassified_id_count = 0

    reader = pd.read_csv(file_path, sep="\t", chunksize=CHUNK_SIZE, dtype=str)
    for chunk in reader:
        total_rows += len(chunk)
        for _, row in chunk.iterrows():
            if match_col:
                all_ids = split_id_list(row.get(match_col))
                if s2_ids_set is not None or s3_ids_set is not None:
                    n_s2 = sum(1 for i in all_ids if s2_ids_set and i in s2_ids_set)
                    n_s3 = sum(1 for i in all_ids if s3_ids_set and i in s3_ids_set)
                    unclassified_id_count += max(len(all_ids) - n_s2 - n_s3, 0)
                else:
                    n_s2 = n_s3 = None  # cannot classify without source id sets
            else:
                s2_ids = split_id_list(row.get(s2_col)) if s2_col else []
                s3_ids = split_id_list(row.get(s3_col)) if s3_col else []
                all_ids = s2_ids + s3_ids
                n_s2 = len(s2_ids)
                n_s3 = len(s3_ids)

            total_matches = len(all_ids)

            match_count_distribution[total_matches] += 1
            if n_s2 is not None:
                s2_match_counts[n_s2] += 1
                s3_match_counts[n_s3] += 1
                if n_s2 > 0 and n_s3 > 0:
                    both_s2_and_s3 += 1
                elif n_s2 > 0:
                    s1_with_only_s2 += 1
                elif n_s3 > 0:
                    s1_with_only_s3 += 1

            if total_matches == 0:
                zero_match += 1
            elif total_matches == 1:
                one_match += 1
            else:
                multi_match += 1

            if total_matches > max_matches_seen:
                max_matches_seen = total_matches
                max_matches_example_id = row.get(s1_col) if s1_col else None

    match_counts_list_keys = list(match_count_distribution.keys())
    mean_matches = (
        sum(n * c for n, c in match_count_distribution.items()) / total_rows if total_rows else 0.0
    )

    warning = None
    if not s1_col or not (match_col or s2_col or s3_col):
        warning = ("Could not confidently detect ground-truth ID/match columns — inspect "
                   "'columns_raw' below and adjust GT_*_CANDIDATES at the top of this script.")
    elif match_col and s2_ids_set is None and s3_ids_set is None:
        warning = ("Found a single combined match column ('matched_entity_ids'-style) but no "
                   "source2/source3-named files under dataset/, so S2-vs-S3 match counts could "
                   "not be computed (only total match counts are reliable). Point "
                   "SOURCE2_FILENAME_HINTS/SOURCE3_FILENAME_HINTS at the correct filenames to "
                   "enable the split.")
    elif match_col and unclassified_id_count > 0:
        warning = (f"{unclassified_id_count:,} matched IDs did not match any known Source-2 or "
                   f"Source-3 ID — check SOURCE2_FILENAME_HINTS/SOURCE3_FILENAME_HINTS and ID "
                   f"column detection.")

    return {
        "columns_detected": {
            "s1_id_column": s1_col,
            "combined_match_column": match_col,
            "s2_match_column": s2_col,
            "s3_match_column": s3_col,
            "s2_s3_split_available": s2_ids_set is not None or s3_ids_set is not None or bool(s2_col or s3_col),
        },
        "columns_raw": columns,
        "total_s1_entities": total_rows,
        "zero_match_entities": zero_match,
        "one_match_entities": one_match,
        "multi_match_entities": multi_match,
        "min_matches": min(match_counts_list_keys) if match_counts_list_keys else 0,
        "max_matches": max(match_counts_list_keys) if match_counts_list_keys else 0,
        "mean_matches": round(mean_matches, 4),
        "match_count_distribution": dict(sorted(match_count_distribution.items())),
        "s2_match_count_distribution": dict(sorted(s2_match_counts.items())) if s2_match_counts else None,
        "s3_match_count_distribution": dict(sorted(s3_match_counts.items())) if s3_match_counts else None,
        "entities_with_both_s2_and_s3": both_s2_and_s3 if s2_match_counts else None,
        "entities_with_only_s2": s1_with_only_s2 if s2_match_counts else None,
        "entities_with_only_s3": s1_with_only_s3 if s2_match_counts else None,
        "max_matches_for_single_entity": max_matches_seen,
        "example_entity_with_max_matches": max_matches_example_id,
        "warning": warning,
    }


def profile_to_json_safe(profile: ColumnFileProfile) -> Dict[str, Any]:
    return {
        "file_name": profile.file_name,
        "file_size": human_size(profile.file_size_bytes),
        "file_size_bytes": profile.file_size_bytes,
        "columns": profile.columns,
        "row_count": profile.row_count,
        "missing_value_counts": profile.missing_counts,
        "id_column_used": profile.id_column_used,
        "unique_id_count": profile.unique_id_count,
        "country_column_used": profile.country_column_used,
        "country_distribution_top20": dict(profile.country_distribution.most_common(20)),
        "country_distribution_num_unique": len(profile.country_distribution),
        "name_column_used": profile.name_column_used,
        "name_length_stats": profile.name_length_stats.summary() if profile.name_length_stats else None,
        "address_column_used": profile.address_column_used,
        "address_length_stats": profile.address_length_stats.summary() if profile.address_length_stats else None,
        "processing_seconds": profile.duration_seconds,
    }


# --------------------------------------------------------------------------
# MARKDOWN REPORT
# --------------------------------------------------------------------------

def build_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Dataset Profile — Business Entity Resolution (Phase 1)\n")
    a(f"Generated: {report['meta']['generated_at']}  \n"
      f"Project root: `{report['meta']['project_root']}`  \n"
      f"Chunk size: {report['meta']['chunk_size']}  \n"
      f"Random seed: {report['meta']['random_seed']}\n")

    a("## A. Dataset overview\n")
    a(f"- Train files found: {len(report['files']['train'])}")
    a(f"- Test files found: {len(report['files']['test'])}\n")

    a("## B. File sizes and row counts\n")
    a("| Split | File | Size | Rows | Columns |")
    a("|---|---|---|---|---|")
    for split in ("train", "test"):
        for f in report["files"][split]:
            a(f"| {split} | {f['file_name']} | {f['file_size']} | {f['row_count']:,} | {len(f['columns'])} |")
    a("")

    a("## C. Column analysis\n")
    for split in ("train", "test"):
        for f in report["files"][split]:
            a(f"**{split}/{f['file_name']}**: `{', '.join(f['columns'])}`  ")
            a(f"Detected — id: `{f['id_column_used']}`, name: `{f['name_column_used']}`, "
              f"address: `{f['address_column_used']}`, country: `{f['country_column_used']}`\n")

    a("## D. Missing-value analysis\n")
    for split in ("train", "test"):
        for f in report["files"][split]:
            if f["missing_value_counts"]:
                a(f"**{split}/{f['file_name']}**")
                for col, cnt in f["missing_value_counts"].items():
                    if cnt:
                        pct = (cnt / f["row_count"] * 100) if f["row_count"] else 0
                        a(f"- `{col}`: {cnt:,} missing ({pct:.2f}%)")
                a("")

    a("## E. Country distribution\n")
    for split in ("train", "test"):
        for f in report["files"][split]:
            if f["country_distribution_top20"]:
                a(f"**{split}/{f['file_name']}** — {f['country_distribution_num_unique']} unique countries seen")
                for country, cnt in f["country_distribution_top20"].items():
                    a(f"- {country}: {cnt:,}")
                a("")

    a("## F. Name statistics (business_name length, characters)\n")
    for split in ("train", "test"):
        for f in report["files"][split]:
            if f["name_length_stats"] and f["name_length_stats"].get("count"):
                s = f["name_length_stats"]
                a(f"**{split}/{f['file_name']}**: mean={s['mean']}, std={s['std']}, "
                  f"min={s['min']}, max={s['max']}, percentiles={s['approx_percentiles']}")
    a("")

    a("## G. Address statistics (business_address length, characters)\n")
    for split in ("train", "test"):
        for f in report["files"][split]:
            if f["address_length_stats"] and f["address_length_stats"].get("count"):
                s = f["address_length_stats"]
                a(f"**{split}/{f['file_name']}**: mean={s['mean']}, std={s['std']}, "
                  f"min={s['min']}, max={s['max']}, percentiles={s['approx_percentiles']}")
    a("")

    gt = report.get("ground_truth")
    if gt:
        a("## H. Ground-truth match cardinality\n")
        def fmt(v):
            return f"{v:,}" if isinstance(v, int) else str(v)

        a(f"- Total Source 1 entities: {gt['total_s1_entities']:,}")
        a(f"- Zero-match entities: {gt['zero_match_entities']:,}")
        a(f"- One-match entities: {gt['one_match_entities']:,}")
        a(f"- Multi-match entities: {gt['multi_match_entities']:,}")
        a(f"- Min / Max / Mean matches per entity: {gt.get('min_matches')} / "
          f"{gt.get('max_matches')} / {gt.get('mean_matches')}")
        a(f"- Max matches for a single S1 entity: {gt['max_matches_for_single_entity']} "
          f"(example id: {gt['example_entity_with_max_matches']})")
        a(f"- Match-count distribution: `{gt['match_count_distribution']}`")
        a(f"- S2 match-count distribution: `{gt['s2_match_count_distribution']}`")
        a(f"- S3 match-count distribution: `{gt['s3_match_count_distribution']}`\n")

        a("## I. Singleton statistics\n")
        a(f"- Entities with only an S2 match: {fmt(gt['entities_with_only_s2'])}")
        a(f"- Entities with only an S3 match: {fmt(gt['entities_with_only_s3'])}")
        a(f"- Entities with both S2 and S3 matches: {fmt(gt['entities_with_both_s2_and_s3'])}\n")
        if gt.get("warning"):
            a(f"> ⚠️ {gt['warning']}\n")

    a("## J. Noise-pattern examples (from deterministic + random samples)\n")
    a("See `dataset_profile.json` → `samples` for the full head/random rows per file. "
      "Manually scan these for: abbreviations (Pvt/Ltd/Corp), punctuation differences, "
      "spelling variants, word-order changes, duplicated words, transliteration-like "
      "differences in names; Rd/Road, St/Street, missing components, reordering, landmark "
      "references, PIN/postal codes, and state/city variation in addresses.\n")

    a("## K. Important observations\n")
    a("_Fill in after reviewing the JSON output — auto-generated placeholder for manual notes._\n")

    a("## L. Risks for false positives\n")
    a("_Given precision-weighted F0.5, note here which noise patterns are most likely to cause "
      "false matches (e.g., common chain/franchise names, generic strip-mall addresses, "
      "near-duplicate names across different countries)._\n")

    a("## M. Recommendations for normalization\n")
    a("_To be filled in for Phase 2 based on the noise patterns observed above._\n")

    a("## N. Recommendations for blocking\n")
    a("_To be filled in for Phase 2 — e.g. candidate blocking keys such as normalized name "
      "prefix, postal code, city, or country, informed by the country distribution and address "
      "statistics above._\n")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main() -> None:
    project_root = find_project_root()
    dataset_dir = project_root / "dataset"

    log(f"Project root resolved to: {project_root}")
    if not dataset_dir.exists():
        log(f"ERROR: no 'dataset' directory found under {project_root}. "
            f"Run this script from the student_resource project root: "
            f"'python src/data_profiler.py' from D:\\amazon-ml\\student_resource")
        sys.exit(1)

    files_by_split = discover_tsv_files(dataset_dir)
    log(f"Discovered {len(files_by_split['train'])} train file(s), "
        f"{len(files_by_split['test'])} test file(s).")

    report: Dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(project_root),
            "chunk_size": CHUNK_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "files": {"train": [], "test": []},
        "samples": {"train": {}, "test": {}},
        "country_examples": {"train": {}, "test": {}},
        "ground_truth": None,
    }

    ground_truth_path: Optional[Path] = None

    for split, paths in files_by_split.items():
        for path in paths:
            log(f"Profiling {split}/{path.name} ({human_size(path.stat().st_size)}) ...")

            if is_ground_truth_file(path):
                ground_truth_path = path
                log(f"  -> detected as ground-truth file, will run cardinality analysis separately")
                # Still capture basic file stats (size, columns, row count) for the overview table.
                basic = profile_entity_file(path)
                report["files"][split].append(profile_to_json_safe(basic))
                continue

            profile = profile_entity_file(path)
            report["files"][split].append(profile_to_json_safe(profile))

            log(f"  -> {profile.row_count:,} rows, {len(profile.columns)} columns, "
                f"{profile.duration_seconds}s")

            samples = deterministic_and_random_sample(path, profile.columns)
            report["samples"][split][path.name] = samples

            country_examples = per_country_examples(
                path, profile.country_column_used, profile.name_column_used, profile.address_column_used
            )
            report["country_examples"][split][path.name] = country_examples

    if ground_truth_path is not None:
        log(f"Running ground-truth cardinality analysis on {ground_truth_path.name} ...")
        report["ground_truth"] = profile_ground_truth(ground_truth_path, dataset_dir)
    else:
        log("No ground-truth file matched GROUND_TRUTH_FILENAME_HINTS "
            "(looked for filenames containing: " + ", ".join(GROUND_TRUTH_FILENAME_HINTS) + "). "
            "If your file is named differently, add its stem to GROUND_TRUTH_FILENAME_HINTS.")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    json_path = reports_dir / "dataset_profile.json"
    md_path = reports_dir / "dataset_profile.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(report))

    log("=" * 70)
    log("PHASE 1 PROFILING COMPLETE")
    log("=" * 70)
    log(f"Files discovered:")
    for split in ("train", "test"):
        for f in report["files"][split]:
            log(f"  [{split}] {f['file_name']}  size={f['file_size']}  rows={f['row_count']:,}  cols={f['columns']}")
    if report["ground_truth"]:
        gt = report["ground_truth"]
        log(f"Ground truth: total={gt['total_s1_entities']:,} zero={gt['zero_match_entities']:,} "
            f"one={gt['one_match_entities']:,} multi={gt['multi_match_entities']:,}")
    log(f"JSON report: {json_path}")
    log(f"Markdown report: {md_path}")
    log("STOP — Phase 1 only. Review the reports before Phase 2 (normalization/blocking/modeling).")


if __name__ == "__main__":
    main()
