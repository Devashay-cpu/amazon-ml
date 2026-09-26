"""
Phase 2 — Blocking

Builds inverted indexes over a normalized source (Source 2 or Source 3)
using MULTIPLE blocking strategies, and retrieves candidate IDs for a
query record by unioning the keys that strategy would produce for it.

No all-pairs comparison anywhere: every lookup is a dict hit on a
precomputed inverted index (O(1) average per strategy).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# A block bigger than this is still used (we prioritize recall first,
# per spec), but is flagged in the report as a likely false-positive
# generator / scalability risk for Phase 3.
BLOCK_SIZE_WARN_THRESHOLD = 5000

# A name token is eligible for the "rare token" blocking strategy only if
# it appears in at most this many distinct records of the source it's
# being indexed against, and at most this fraction of the source, whichever
# is smaller. Filters out common words that would otherwise create huge
# blocks (e.g. "group", "international", "the").
RARE_TOKEN_MAX_DOC_FRACTION = 0.0005
RARE_TOKEN_MAX_DOC_CAP = 50
RARE_TOKEN_MIN_LEN = 3

STRATEGY_NAMES = [
    "country_exact_name",
    "country_name_prefix4",
    "country_suffix_stripped_name",
    "country_rare_name_token",
    "postal_code",
    "house_number_street",
    "address_alnum_prefix8",
]


def generate_keys(norm_name: Dict[str, Any], norm_addr: Dict[str, Any],
                   norm_country: Dict[str, Any], token_doc_freq: Optional[Counter],
                   rare_token_threshold: Optional[int]) -> List[Tuple[str, str]]:
    """Return the list of (strategy_name, block_key) pairs this record maps to."""
    keys: List[Tuple[str, str]] = []
    country = norm_country["normalized"] if norm_country else ""

    if norm_name.get("exact_normalized") and country:
        keys.append(("country_exact_name", f"{country}|{norm_name['exact_normalized']}"))

    if norm_name.get("prefix4") and country:
        keys.append(("country_name_prefix4", f"{country}|{norm_name['prefix4']}"))

    if norm_name.get("suffix_stripped") and country:
        keys.append(("country_suffix_stripped_name", f"{country}|{norm_name['suffix_stripped']}"))

    if token_doc_freq is not None and rare_token_threshold is not None:
        for tok in set(norm_name.get("tokens", [])):
            if len(tok) >= RARE_TOKEN_MIN_LEN and token_doc_freq.get(tok, 0) <= rare_token_threshold:
                keys.append(("country_rare_name_token", f"{country}|{tok}"))

    if norm_addr:
        if norm_addr.get("postal_code"):
            keys.append(("postal_code", f"{country}|{norm_addr['postal_code']}"))

        if norm_addr.get("house_number") and norm_addr.get("street_tokens"):
            street_key = norm_addr["street_tokens"][0]
            keys.append(("house_number_street", f"{country}|{norm_addr['house_number']}|{street_key}"))

        if norm_addr.get("address_alnum"):
            keys.append(("address_alnum_prefix8", f"{country}|{norm_addr['address_alnum'][:8]}"))

    return keys


class BlockIndex:
    """
    Two-pass inverted index builder for one source (S2 or S3):
      pass 1 (add_pass1): counts name-token document frequency, needed to
        decide which tokens are "rare" enough to use as a blocking key.
      finalize_pass1: fixes the rarity threshold.
      pass 2 (add_pass2): builds the actual strategy -> key -> [ids] indexes.
    """

    def __init__(self, label: str):
        self.label = label
        self.token_doc_freq: Counter = Counter()
        self.total_docs = 0
        self.rare_token_threshold: Optional[int] = None
        self.indices: Dict[str, Dict[str, List[str]]] = {s: defaultdict(list) for s in STRATEGY_NAMES}
        self.all_ids: Set[str] = set()

    def add_pass1(self, norm_name: Dict[str, Any]) -> None:
        self.total_docs += 1
        for tok in set(norm_name.get("tokens", [])):
            self.token_doc_freq[tok] += 1

    def finalize_pass1(self) -> None:
        self.rare_token_threshold = max(
            1, min(RARE_TOKEN_MAX_DOC_CAP, int(self.total_docs * RARE_TOKEN_MAX_DOC_FRACTION))
        )

    def add_pass2(self, entity_id: str, norm_name: Dict[str, Any],
                  norm_addr: Dict[str, Any], norm_country: Dict[str, Any]) -> None:
        self.all_ids.add(entity_id)
        keys = generate_keys(norm_name, norm_addr, norm_country,
                              self.token_doc_freq, self.rare_token_threshold)
        for strategy, key in keys:
            self.indices[strategy][key].append(entity_id)

    def get_candidates(self, norm_name: Dict[str, Any], norm_addr: Dict[str, Any],
                        norm_country: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """Return (union of candidate ids, {strategy: ids from that strategy})."""
        keys = generate_keys(norm_name, norm_addr, norm_country,
                              self.token_doc_freq, self.rare_token_threshold)
        per_strategy: Dict[str, Set[str]] = {}
        union: Set[str] = set()
        for strategy, key in keys:
            ids = self.indices[strategy].get(key)
            if ids:
                id_set = set(ids)
                per_strategy.setdefault(strategy, set()).update(id_set)
                union.update(id_set)
        return union, per_strategy

    def problematic_blocks(self, top_n: int = 20) -> List[Tuple[str, str, int]]:
        """(strategy, key, size) for the largest blocks, largest first."""
        rows: List[Tuple[str, str, int]] = []
        for strategy, d in self.indices.items():
            for key, ids in d.items():
                if len(ids) > BLOCK_SIZE_WARN_THRESHOLD:
                    rows.append((strategy, key, len(ids)))
        rows.sort(key=lambda r: -r[2])
        return rows[:top_n]

    def strategy_key_counts(self) -> Dict[str, int]:
        return {s: len(d) for s, d in self.indices.items()}
