"""
Phase 7 — Model inference module

Loads the trained Phase 5 pipeline (persisted by train_model_v2.py) and the
Phase 6 calibrated threshold (persisted by phase6_pipeline.py), and scores
new candidate/reference record pairs using EXACTLY the same preprocessing
as training: normalize_business_name / normalize_address / normalize_country
(Phase 1/3, unchanged) -> features.extract_pair_features (Phase 3,
unchanged, same 16-feature schema) -> the persisted Phase 5 pipeline's
predict_proba (Phase 5, unchanged).

No training happens anywhere in this module. If the persisted model or
metadata is missing, or an input record is missing a required field, this
module raises a clear, specific exception rather than silently guessing.

Usage:
    from inference import load_model
    model = load_model()  # loads models/phase5_model.joblib etc.
    result = model.predict_pair(
        candidate={"entity_id": "s2_1", "business_name": "Acme Corp",
                   "business_address": "123 Main St 94105", "country": "USA"},
        reference={"entity_id": "s1_1", "business_name": "Acme Corporation",
                   "business_address": "123 Main Street 94105", "country": "USA"},
    )
    # result.score, result.prediction, result.threshold
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from features import FEATURE_NAMES, extract_pair_features
from normalization import normalize_address, normalize_business_name, normalize_country
from train_baseline import find_project_root  # Phase 4, reused unchanged

REQUIRED_RECORD_FIELDS = ["business_name", "business_address", "country"]


class InvalidRecordError(ValueError):
    """Raised when a candidate/reference record is structurally invalid
    (not a dict, or missing a required field key). Note: an EMPTY value for
    a required field (e.g. business_name="") is valid input -- normalize_*
    already handles that gracefully and produces the MISSING(-1) feature
    sentinel, same as during training. This error is only for a record that
    is malformed at the structural level (wrong type, or the key absent
    entirely), which is a client-side bug, not ordinary missing data."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when the persisted Phase 5 model/metadata cannot be found on disk."""


class SchemaMismatchError(ValueError):
    """Raised when a loaded model's recorded feature schema no longer
    matches the current features.FEATURE_NAMES -- signals the model was
    trained against a different feature definition and must be retrained,
    rather than silently scoring with a misaligned feature vector."""


@dataclass
class PredictionResult:
    score: float
    prediction: int
    threshold: float
    candidate_entity_id: Optional[str] = None
    reference_entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_record(record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise InvalidRecordError(f"'{label}' must be a JSON object (dict), got {type(record).__name__}")
    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
    if missing:
        raise InvalidRecordError(
            f"'{label}' record is missing required field(s): {missing}. "
            f"Required fields: {REQUIRED_RECORD_FIELDS} (values may be empty strings, but the keys "
            f"must be present)."
        )


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Same three normalization calls Phase 3 uses to build a record for
    features.extract_pair_features -- no new preprocessing is introduced."""
    return {
        "name": normalize_business_name(record.get("business_name")),
        "address": normalize_address(record.get("business_address")),
        "country": normalize_country(record.get("country")),
    }


class InferenceModel:
    """
    Thin wrapper around a fitted sklearn Pipeline plus its recorded feature
    schema and calibrated decision threshold. Constructed by load_model()
    below -- not intended to be constructed directly with an unvalidated
    pipeline, since the schema check that guards against a stale/mismatched
    model happens in __init__.
    """

    def __init__(self, pipeline, feature_columns: List[str], threshold: float, metadata: Dict[str, Any]):
        if list(feature_columns) != list(FEATURE_NAMES):
            raise SchemaMismatchError(
                "Loaded model's recorded feature schema does not match the current Phase 3 "
                f"FEATURE_NAMES.\n  Loaded:  {feature_columns}\n  Current: {list(FEATURE_NAMES)}\n"
                "This means features.py changed since this model was trained. Retrain via "
                "'python src/train_model_v2.py' before using this model for inference."
            )
        self._pipeline = pipeline
        self.feature_columns = list(feature_columns)
        self.threshold = float(threshold)
        self.metadata = metadata

    def _features_to_vector(self, feats: Dict[str, Any]) -> np.ndarray:
        return np.array([[feats[name] for name in self.feature_columns]], dtype=np.float32)

    def predict_pair(self, candidate: Dict[str, Any], reference: Dict[str, Any]) -> PredictionResult:
        """
        candidate / reference: raw record dicts with business_name,
        business_address, country (entity_id optional, kept only for
        traceability in the returned result -- never used as a feature,
        exactly as in Phase 3).
        """
        _validate_record(candidate, "candidate")
        _validate_record(reference, "reference")

        cand_norm = _normalize_record(candidate)
        ref_norm = _normalize_record(reference)
        # Matches Phase 3's extract_pair_features(s1_record, cand_record)
        # convention: "reference" plays the role of the fixed Source-1
        # entity, "candidate" plays the role of the S2/S3 candidate.
        feats = extract_pair_features(ref_norm, cand_norm)

        X = self._features_to_vector(feats)
        proba = float(self._pipeline.predict_proba(X)[0, 1])
        prediction = int(proba >= self.threshold)

        return PredictionResult(
            score=round(proba, 6),
            prediction=prediction,
            threshold=self.threshold,
            candidate_entity_id=candidate.get("entity_id"),
            reference_entity_id=reference.get("entity_id"),
        )

    def predict_batch(
        self, pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[PredictionResult]:
        """pairs: list of (candidate, reference) dict tuples. Each pair is
        validated and scored independently -- one invalid pair raises
        immediately (fail clearly) rather than silently skipping it."""
        return [self.predict_pair(candidate, reference) for candidate, reference in pairs]


def _default_model_dir() -> Path:
    return find_project_root() / "models"


def load_model(model_dir: Optional[Path] = None) -> InferenceModel:
    """
    Loads the persisted Phase 5 pipeline (models/phase5_model.joblib) and
    its metadata (models/phase5_model_metadata.json), and applies the
    Phase 6 calibrated threshold if models/phase6_threshold.json is present
    -- otherwise falls back to Phase 5's own selected threshold, recorded
    in metadata["threshold_source"] either way so callers can tell which
    one is in effect. Never trains anything.
    """
    import joblib  # local import: only needed when actually loading a model

    resolved_dir = Path(model_dir) if model_dir else _default_model_dir()
    model_path = resolved_dir / "phase5_model.joblib"
    metadata_path = resolved_dir / "phase5_model_metadata.json"
    threshold_path = resolved_dir / "phase6_threshold.json"

    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotFoundError(
            f"No trained model found under {resolved_dir}. Run 'python src/train_model_v2.py' first "
            f"-- it saves {model_path.name} and {metadata_path.name} there."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = dict(json.load(f))

    threshold = metadata.get("phase5_selected_threshold", metadata.get("default_threshold", 0.5))
    threshold_source = "phase5_max_f1"

    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_meta = json.load(f)
        threshold = threshold_meta.get("selected_threshold", threshold)
        threshold_source = "phase6_cost_calibrated"
        metadata["phase6_threshold_metadata"] = threshold_meta
    else:
        metadata.setdefault(
            "phase6_threshold_note",
            "models/phase6_threshold.json not found -- using Phase 5's own selected threshold "
            "instead of a Phase 6 cost-calibrated one. Run src/phase6_pipeline.py to calibrate one.",
        )

    metadata["threshold_source"] = threshold_source
    return InferenceModel(pipeline, metadata["feature_columns"], threshold, metadata)
