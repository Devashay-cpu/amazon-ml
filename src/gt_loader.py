"""
Shared ground-truth loader — reuses the verified Phase-1 parsing fix
(combined `matched_entity_ids` column, split by comma/semicolon/pipe, then
classified into S2 vs S3 by ID-set membership).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

CHUNK_SIZE = 100_000

S1_ID_CANDIDATES = ["source1_entity_id", "s1_id", "source1_id", "s1", "id", "entity_id"]
MATCH_COLUMN_CANDIDATES = ["matched_entity_ids", "matched_ids", "matches", "match_ids", "matching_ids"]
LIST_SPLIT_CHARS = [";", "|", ","]


def find_column(columns: List[str], candidates: List[str]):
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


def load_ground_truth(gt_path: Path, s2_ids: Set[str], s3_ids: Set[str]) -> Dict[str, Dict[str, Set[str]]]:
    """
    Returns {s1_entity_id: {"s2": set(matched s2 ids), "s3": set(matched s3 ids)}}.
    IDs in matched_entity_ids that aren't found in either s2_ids or s3_ids
    are counted (and can be inspected) but not silently dropped.
    """
    header_cols = pd.read_csv(gt_path, sep="\t", dtype=str, nrows=0).columns.tolist()
    s1_col = find_column(header_cols, S1_ID_CANDIDATES)
    match_col = find_column(header_cols, MATCH_COLUMN_CANDIDATES)
    if s1_col is None or match_col is None:
        raise ValueError(
            f"Could not detect ground-truth columns from header {header_cols}. "
            f"Adjust S1_ID_CANDIDATES/MATCH_COLUMN_CANDIDATES in gt_loader.py."
        )

    result: Dict[str, Dict[str, Set[str]]] = {}
    unclassified = 0

    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            s1_id = str(row_d.get(s1_col))
            all_ids = split_id_list(row_d.get(match_col))
            s2_matched = {i for i in all_ids if i in s2_ids}
            s3_matched = {i for i in all_ids if i in s3_ids}
            unclassified += max(len(all_ids) - len(s2_matched) - len(s3_matched), 0)
            result[s1_id] = {"s2": s2_matched, "s3": s3_matched}

    if unclassified:
        print(f"[gt_loader] WARNING: {unclassified:,} matched IDs did not match any known "
              f"S2/S3 ID — check ID normalization/dtype consistency between files.")

    return result
