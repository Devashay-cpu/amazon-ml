"""
Phase 7 — REST API

Thin FastAPI wrapper around src/inference.py (model scoring) and
src/ranking.py (sorting). The actual request-handling logic lives in the
plain functions below (handle_health/handle_predict/handle_rank), which
take/return plain dicts and raise ApiError on invalid input. These
functions have NO dependency on FastAPI/pydantic and are fully unit
-testable on their own (see tests/test_api.py); the FastAPI app defined at
the bottom of this file is a thin translation layer that calls them and
maps ApiError -> HTTPException.

FastAPI/pydantic/uvicorn are new dependencies for this project (Phase 1-6
used only pandas/numpy/scikit-learn/joblib) -- introduced here because the
Phase 7 spec explicitly asks for a REST API and prefers FastAPI. No other
web framework or ORM is added.

Run locally (from the project root, after `pip install fastapi uvicorn`):
    uvicorn src.api:app --reload

Then, in another terminal:
    curl http://127.0.0.1:8000/health

    curl -X POST http://127.0.0.1:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
            "candidate": {"entity_id": "s2_1", "business_name": "Acme Corp",
                           "business_address": "123 Main St 94105", "country": "USA"},
            "reference": {"entity_id": "s1_1", "business_name": "Acme Corporation",
                           "business_address": "123 Main Street 94105", "country": "USA"}
          }'

    curl -X POST http://127.0.0.1:8000/rank \\
      -H "Content-Type: application/json" \\
      -d '{"pairs": [ {"candidate": {...}, "reference": {...}}, ... ], "top_k": 5}'
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from inference import InvalidRecordError, ModelNotFoundError, SchemaMismatchError, load_model
from ranking import InvalidScoredPairError, rank_candidates

# --------------------------------------------------------------------------
# Framework-agnostic request handling (no FastAPI/pydantic dependency here)
# --------------------------------------------------------------------------


class ApiError(Exception):
    """Carries an HTTP status code + a safe, user-facing message. Never
    carries a stack trace or internal exception detail -- callers (the
    FastAPI layer below) surface only .message to the client."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


_model_cache: Dict[str, Any] = {}


def get_model():
    """
    Loads the model once and caches it for the life of the process -- no
    retraining, and no re-loading from disk on every request. A missing
    model surfaces as a clear 503 (service not ready), not a 500.
    """
    if "model" not in _model_cache:
        try:
            _model_cache["model"] = load_model()
        except ModelNotFoundError as e:
            raise ApiError(503, str(e)) from e
        except SchemaMismatchError as e:
            raise ApiError(503, str(e)) from e
    return _model_cache["model"]


def reset_model_cache() -> None:
    """Test-only helper: clears the cached model so tests can control
    exactly when/whether load_model() is invoked."""
    _model_cache.pop("model", None)


def handle_health() -> Dict[str, Any]:
    """GET /health. Never raises -- a model-loading problem is reported as
    a degraded (but 200) health status, since /health itself must always
    respond, even when the model isn't ready yet."""
    try:
        model = get_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "threshold": model.threshold,
            "threshold_source": model.metadata.get("threshold_source"),
            "feature_count": len(model.feature_columns),
        }
    except ApiError as e:
        return {"status": "degraded", "model_loaded": False, "detail": e.message}


def handle_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /predict body: {"candidate": {...}, "reference": {...}}."""
    if not isinstance(payload, dict):
        raise ApiError(400, "Request body must be a JSON object")

    candidate = payload.get("candidate")
    reference = payload.get("reference")
    if candidate is None or reference is None:
        raise ApiError(400, "Request body must include both 'candidate' and 'reference' objects")

    model = get_model()
    try:
        result = model.predict_pair(candidate, reference)
    except InvalidRecordError as e:
        raise ApiError(400, str(e)) from e

    return {
        "score": result.score,
        "prediction": result.prediction,
        "threshold": result.threshold,
        "candidate_entity_id": result.candidate_entity_id,
        "reference_entity_id": result.reference_entity_id,
    }


def handle_rank(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /rank body: {"pairs": [{"candidate": {...}, "reference": {...}}, ...], "top_k": int|null}."""
    if not isinstance(payload, dict):
        raise ApiError(400, "Request body must be a JSON object")

    pairs = payload.get("pairs")
    top_k = payload.get("top_k")
    if not isinstance(pairs, list) or not pairs:
        raise ApiError(400, "Request body must include a non-empty 'pairs' list")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ApiError(400, "'top_k' must be a non-negative integer or null")

    model = get_model()
    scored: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        candidate = pair.get("candidate") if isinstance(pair, dict) else None
        reference = pair.get("reference") if isinstance(pair, dict) else None
        if candidate is None or reference is None:
            raise ApiError(400, f"pairs[{i}] must include both 'candidate' and 'reference'")
        try:
            result = model.predict_pair(candidate, reference)
        except InvalidRecordError as e:
            raise ApiError(400, f"pairs[{i}]: {e}") from e
        scored.append({
            "candidate_entity_id": result.candidate_entity_id,
            "reference_entity_id": result.reference_entity_id,
            "score": result.score,
            "prediction": result.prediction,
        })

    try:
        ranked = rank_candidates(scored, top_k=top_k)
    except InvalidScoredPairError as e:
        raise ApiError(400, str(e)) from e

    return {"results": [r.to_dict() for r in ranked], "count": len(ranked)}


# --------------------------------------------------------------------------
# FastAPI app (thin layer). Optional import: if fastapi/pydantic aren't
# installed, `app` is left as None, and everything above remains usable
# and independently testable without them.
# --------------------------------------------------------------------------

app = None

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Entity Matching Inference API", version="phase7")

    class PredictRequest(BaseModel):
        candidate: Dict[str, Any]
        reference: Dict[str, Any]

    class RankPairRequest(BaseModel):
        candidate: Dict[str, Any]
        reference: Dict[str, Any]

    class RankRequest(BaseModel):
        pairs: List[Dict[str, Any]]
        top_k: Optional[int] = None

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return handle_health()

    @app.post("/predict")
    def predict(request: PredictRequest) -> Dict[str, Any]:
        try:
            return handle_predict(request.model_dump())
        except ApiError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

    @app.post("/rank")
    def rank(request: RankRequest) -> Dict[str, Any]:
        try:
            return handle_rank(request.model_dump())
        except ApiError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never expose internal stack traces / exception details in normal
        # API responses -- log server-side if desired, return a generic
        # message to the client.
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

except ImportError:
    pass  # fastapi/pydantic not installed here; handle_* functions above remain fully usable/testable.
