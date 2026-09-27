"""
Phase 7 — Ranking layer

Given a list of already-scored candidate pairs (each a dict with at least
"score"), sorts them by score descending, assigns 1-based ranks, and
supports top-k truncation. This module does NOT call the model itself and
introduces no new ML model -- it operates purely on scores/identifiers
already produced by src/inference.py, keeping ranking logic decoupled from
model inference per spec.

Determinism: equal scores are broken by candidate_entity_id ascending
(falling back to an empty string if absent), not by insertion order, which
could vary between calls/processes -- so the same input always produces
the same output order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


class InvalidScoredPairError(ValueError):
    """Raised when an item passed to rank_candidates isn't a dict with a
    numeric 'score' key -- a structural input error, fails clearly."""


@dataclass
class RankedResult:
    rank: int
    candidate_entity_id: Optional[str]
    reference_entity_id: Optional[str]
    score: float
    prediction: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rank_candidates(scored_pairs: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[RankedResult]:
    """
    scored_pairs: list of dicts, each with at least "score" (float);
    "prediction" (0/1, default 0 if absent), "candidate_entity_id", and
    "reference_entity_id" are optional and preserved into the output.
    top_k: if given, only the top_k highest-scoring results are returned
    (still fully sorted first, then truncated -- never truncated before
    sorting).
    """
    if not isinstance(scored_pairs, list):
        raise InvalidScoredPairError(f"scored_pairs must be a list, got {type(scored_pairs).__name__}")

    for i, pair in enumerate(scored_pairs):
        if not isinstance(pair, dict):
            raise InvalidScoredPairError(f"scored_pairs[{i}] must be a dict, got {type(pair).__name__}")
        if "score" not in pair:
            raise InvalidScoredPairError(f"scored_pairs[{i}] is missing required key 'score'")
        try:
            float(pair["score"])
        except (TypeError, ValueError) as e:
            raise InvalidScoredPairError(f"scored_pairs[{i}]['score'] is not numeric: {pair['score']!r}") from e

    if top_k is not None and top_k < 0:
        raise InvalidScoredPairError(f"top_k must be >= 0, got {top_k}")

    def sort_key(pair: Dict[str, Any]):
        candidate_id = str(pair.get("candidate_entity_id") or "")
        return (-float(pair["score"]), candidate_id)

    ordered = sorted(scored_pairs, key=sort_key)
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        RankedResult(
            rank=i,
            candidate_entity_id=pair.get("candidate_entity_id"),
            reference_entity_id=pair.get("reference_entity_id"),
            score=float(pair["score"]),
            prediction=int(pair.get("prediction", 0)),
        )
        for i, pair in enumerate(ordered, start=1)
    ]
