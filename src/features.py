"""
Phase 3 — Pair-level feature extraction

Builds a deterministic similarity feature vector for one (Source-1 record,
candidate record) pair, using ONLY the fields that actually exist in this
dataset: business_name, business_address, country (via their normalized
representations from normalization.py). No fields are invented.

MISSING-VALUE HANDLING (data-quality requirement):
Every feature that compares a value on each side uses MISSING (-1) instead
of silently treating "both blank" as a match (1) or "one blank" as a
mismatch (0). Confusing "unknown" with "confirmed non-match" would bias
any model trained on this data, so it's kept explicit and documented here
rather than swept into 0/1.

DETERMINISM: extract_pair_features has no random state, no I/O, and no
mutable module-level state that changes between calls — same two normalized
records always produce the same feature dict.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Optional

MISSING = -1  # sentinel for "cannot compare — one or both sides had no value"

FEATURE_NAMES = [
    "name_exact_match",
    "name_suffix_stripped_match",
    "name_alnum_match",
    "name_token_sorted_match",
    "name_prefix4_match",
    "name_char_similarity",
    "name_token_jaccard",
    "name_token_count_diff",
    "address_exact_match",
    "address_alnum_match",
    "address_char_similarity",
    "house_number_match",
    "postal_code_match",
    "unit_token_match",
    "street_token_jaccard",
    "country_match",
]


def _exact_match_or_missing(a: Optional[str], b: Optional[str]) -> int:
    if not a or not b:
        return MISSING
    return 1 if a == b else 0


def _char_similarity_or_missing(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return MISSING
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def _jaccard_or_missing(a, b) -> float:
    set_a, set_b = set(a or []), set(b or [])
    if not set_a or not set_b:
        return MISSING
    union = set_a | set_b
    if not union:
        return MISSING
    return round(len(set_a & set_b) / len(union), 4)


def extract_pair_features(s1_record: Dict[str, Any], cand_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    s1_record / cand_record: dicts shaped like
        {"name": normalize_business_name(...) output,
         "address": normalize_address(...) output,
         "country": normalize_country(...) output}
    as produced by candidate_generation.iter_candidates (for S1, per-row)
    and candidate_generation.build_record_lookup (for S2/S3 candidates).

    Returns a flat dict with exactly the keys in FEATURE_NAMES, in that
    order, so downstream consumers (CSV writer, tests) get a stable schema.
    """
    n1, n2 = s1_record.get("name", {}), cand_record.get("name", {})
    a1, a2 = s1_record.get("address", {}), cand_record.get("address", {})
    c1, c2 = s1_record.get("country", {}), cand_record.get("country", {})

    features = {
        "name_exact_match": _exact_match_or_missing(n1.get("exact_normalized"), n2.get("exact_normalized")),
        "name_suffix_stripped_match": _exact_match_or_missing(n1.get("suffix_stripped"), n2.get("suffix_stripped")),
        "name_alnum_match": _exact_match_or_missing(n1.get("alnum_normalized"), n2.get("alnum_normalized")),
        "name_token_sorted_match": _exact_match_or_missing(n1.get("token_sorted"), n2.get("token_sorted")),
        "name_prefix4_match": _exact_match_or_missing(n1.get("prefix4"), n2.get("prefix4")),
        "name_char_similarity": _char_similarity_or_missing(n1.get("exact_normalized"), n2.get("exact_normalized")),
        "name_token_jaccard": _jaccard_or_missing(n1.get("tokens"), n2.get("tokens")),
        "name_token_count_diff": abs(len(n1.get("tokens") or []) - len(n2.get("tokens") or [])),
        "address_exact_match": _exact_match_or_missing(a1.get("address_normalized"), a2.get("address_normalized")),
        "address_alnum_match": _exact_match_or_missing(a1.get("address_alnum"), a2.get("address_alnum")),
        "address_char_similarity": _char_similarity_or_missing(a1.get("address_normalized"), a2.get("address_normalized")),
        "house_number_match": _exact_match_or_missing(a1.get("house_number"), a2.get("house_number")),
        "postal_code_match": _exact_match_or_missing(a1.get("postal_code"), a2.get("postal_code")),
        "unit_token_match": _exact_match_or_missing(a1.get("unit_token"), a2.get("unit_token")),
        "street_token_jaccard": _jaccard_or_missing(a1.get("street_tokens"), a2.get("street_tokens")),
        "country_match": _exact_match_or_missing(c1.get("normalized"), c2.get("normalized")),
    }
    # Defensive ordering guarantee (schema stability), not a behavior change.
    return {name: features[name] for name in FEATURE_NAMES}
