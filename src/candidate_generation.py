"""
Phase 2 — Candidate generation

Streams a Source-1 file and, for each row, produces candidate ID sets from
Source-2 and Source-3 inverted indexes (built by blocking.BlockIndex).

Everything here is chunked/streaming: no full-file loads, no all-pairs
comparison, no giant in-memory pair DataFrame. Candidates are yielded one
Source-1 entity at a time so callers (e.g. evaluate_blocking.py) can
accumulate statistics without holding every candidate set in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import pandas as pd

from blocking import BlockIndex
from normalization import normalize_address, normalize_business_name, normalize_country

CHUNK_SIZE = 100_000

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def build_index_from_source(file_path: Path, label: str) -> BlockIndex:
    """
    Two-pass build: pass 1 only reads entity_id + business_name (to compute
    token document frequency for the rare-token strategy); pass 2 re-reads
    the file and builds the full inverted index. Re-reading is a deliberate
    memory/CPU tradeoff: it avoids holding every normalized record from
    pass 1 in memory between passes.
    """
    idx = BlockIndex(label)

    for chunk in pd.read_csv(file_path, sep="\t", usecols=["business_name"],
                              dtype=str, chunksize=CHUNK_SIZE):
        for name in chunk["business_name"].tolist():
            idx.add_pass1(normalize_business_name(name))
    idx.finalize_pass1()

    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            norm_name = normalize_business_name(row_d.get("business_name"))
            norm_addr = normalize_address(row_d.get("business_address"))
            norm_country = normalize_country(row_d.get("country"))
            idx.add_pass2(str(row_d.get("entity_id")), norm_name, norm_addr, norm_country)

    return idx


def build_record_lookup(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Phase 3 addition (additive only — does not change build_index_from_source
    or iter_candidates above). Streams a source file once and returns
    {entity_id: {"name": <normalize_business_name output>,
                 "address": <normalize_address output>,
                 "country": <normalize_country output>}}
    so pair-level feature extraction can look up a candidate's normalized
    fields by ID after blocking has already selected which IDs matter.

    This holds one normalized record per row in memory — necessary because
    feature extraction needs random access by ID, unlike index building
    (which only needs streaming access). For the full ~2.2M-row sources this
    is the memory-heavy step of Phase 3; see docs/PHASE3_DESIGN.md for the
    scaling note (e.g. restricting this to only IDs that actually appear in
    at least one candidate set, rather than all of S2/S3).
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            entity_id = str(row_d.get("entity_id"))
            lookup[entity_id] = {
                "name": normalize_business_name(row_d.get("business_name")),
                "address": normalize_address(row_d.get("business_address")),
                "country": normalize_country(row_d.get("country")),
            }
    return lookup


def build_record_lookup_for_ids(file_path: Path, ids_needed: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Same as build_record_lookup, but only retains records whose entity_id is
    in `ids_needed` — the memory-saving variant recommended in
    docs/PHASE3_DESIGN.md once you know which candidate IDs actually matter
    (e.g. the union of all candidate sets a blocking run produced).
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            entity_id = str(row_d.get("entity_id"))
            if entity_id in ids_needed:
                lookup[entity_id] = {
                    "name": normalize_business_name(row_d.get("business_name")),
                    "address": normalize_address(row_d.get("business_address")),
                    "country": normalize_country(row_d.get("country")),
                }
    return lookup


def iter_candidates(
    s1_file_path: Path,
    s2_index: Optional[BlockIndex],
    s3_index: Optional[BlockIndex],
    limit: Optional[int] = None,
) -> Iterator[Tuple[str, Set[str], Set[str], dict, dict]]:
    """
    Yields (entity_id, s2_candidate_ids, s3_candidate_ids,
    s2_per_strategy_ids, s3_per_strategy_ids) for each Source-1 row,
    streaming from disk. Stops after `limit` rows if given.
    """
    seen = 0
    for chunk in pd.read_csv(s1_file_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            if limit is not None and seen >= limit:
                return
            row_d = row._asdict()
            entity_id = str(row_d.get("entity_id"))
            norm_name = normalize_business_name(row_d.get("business_name"))
            norm_addr = normalize_address(row_d.get("business_address"))
            norm_country = normalize_country(row_d.get("country"))

            s2_cands: Set[str] = set()
            s2_per_strategy: dict = {}
            if s2_index is not None:
                s2_cands, s2_per_strategy = s2_index.get_candidates(norm_name, norm_addr, norm_country)

            s3_cands: Set[str] = set()
            s3_per_strategy: dict = {}
            if s3_index is not None:
                s3_cands, s3_per_strategy = s3_index.get_candidates(norm_name, norm_addr, norm_country)

            yield entity_id, s2_cands, s3_cands, s2_per_strategy, s3_per_strategy
            seen += 1
