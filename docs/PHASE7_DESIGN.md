# Phase 7 Design — Inference API + Ranking Layer

## Architecture

```
Raw candidate/reference records (JSON)
  -> normalization.py                    (Phase 1/3, unchanged)
  -> features.extract_pair_features      (Phase 3, unchanged, same 16-feature schema)
  -> inference.InferenceModel            (Phase 7, NEW — loads models/phase5_model.joblib)
       .predict_pair() / .predict_batch()
  -> ranking.rank_candidates             (Phase 7, NEW — sorts scored pairs, no new model)
  -> api.py                              (Phase 7, NEW — thin FastAPI layer over the above)
```

`src/inference.py` and `src/ranking.py` have **no FastAPI dependency** — they are plain
Python modules usable from any script, notebook, or test. `src/api.py` is a thin HTTP
wrapper around them; its request-handling logic (`handle_health`/`handle_predict`/`handle_rank`)
is itself framework-agnostic (plain dict in, plain dict out, raises `ApiError`), and the
FastAPI routes at the bottom of the file are a few lines of translation on top.

## Model loading (no training in this phase)

Phase 5/6 didn't previously persist a trained model to disk — `train_model_v2.py` fit and
evaluated the pipeline in memory each run. Phase 7 required a persisted artifact to load, so
two **minimal, additive** changes were made (nothing existing was removed or altered):

- `train_model_v2.py`: after fitting and evaluating the Phase 5 model exactly as before, it now
  also saves the fitted pipeline to `models/phase5_model.joblib` and its schema/metadata to
  `models/phase5_model_metadata.json` (opt out with `--no-save-model`). No metric, split, or
  report content changed.
- `phase6_pipeline.py`: after computing the cost-selected threshold exactly as before, it now
  also saves it to `models/phase6_threshold.json` (opt out with `--no-save-threshold`).

A real cross-module pickling bug was found and fixed during integration: when
`train_model_v2.py` is run directly (`python train_model_v2.py`), Python executes it as module
`"__main__"`, so a joblib-pickled `FunctionTransformer(_sentinel_to_nan)` recorded its callable's
module as `"__main__"` — unloadable from any other script. Fixed with
`sys.modules.setdefault("train_model_v2", sys.modules[__name__])`, which aliases the
currently-running module under its real importable name so pickle's identity check succeeds
regardless of how the script was invoked. This changes no runtime behavior, only cross-script
persistence.

`inference.load_model()` loads both files (from `models/` by default, resolved via the same
`find_project_root()` pattern every other script in this repo uses), validates that the loaded
model's recorded `feature_columns` still matches `features.FEATURE_NAMES` exactly (raising
`SchemaMismatchError` if not — guards against silently scoring with a stale/misaligned feature
vector after a future features.py change), and returns an `InferenceModel`. No training ever
happens in this path — `joblib.load` only.

## Feature compatibility

`InferenceModel.predict_pair(candidate, reference)` calls the exact same three normalization
functions and `features.extract_pair_features` that Phase 3 uses to build the training dataset —
imported unchanged, never reimplemented. The convention: `reference` plays the role of the fixed
Source-1 entity, `candidate` plays the role of the S2/S3 candidate, matching
`extract_pair_features(s1_record, cand_record)`'s parameter order from Phase 3.

Required input fields (matching the dataset schema exactly, no invented fields):
`business_name`, `business_address`, `country`. An **empty value** for these (e.g.
`business_name=""`) is valid input — `normalize_*` already handles it, producing the same
`MISSING=-1` feature sentinel used during training. A **missing key** (the field absent from the
JSON entirely) is treated as a client-side error and raises `InvalidRecordError` — "fail clearly
for missing or invalid required features," per spec.

## Phase 6 threshold usage

`load_model()` prefers `models/phase6_threshold.json` (the cost-calibrated threshold) over Phase
5's own max-F1 threshold if present, and records which one is in effect in
`metadata["threshold_source"]` (`"phase6_cost_calibrated"` or `"phase5_max_f1"`), exposed via
`GET /health`. The threshold is never silently changed by Phase 7 — it only reads whatever Phase
5/6 already computed and persisted.

## Ranking logic

`rank_candidates(scored_pairs, top_k=None)` is a pure function over already-scored pairs (each a
dict with at least `"score"`). It never calls the model — ranking is deliberately decoupled from
inference (per spec, "keep ranking separate from model inference"; "do not introduce a new ML
model"). Sorting is descending by score; ties are broken by `candidate_entity_id` ascending
(not insertion order), so identical input always produces identical output regardless of
call-to-call ordering variance.

## API endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status", "model_loaded", "threshold", "threshold_source", "feature_count"}` |
| POST | `/predict` | `{"candidate": {...}, "reference": {...}}` | `{"score", "prediction", "threshold", "candidate_entity_id", "reference_entity_id"}` |
| POST | `/rank` | `{"pairs": [{"candidate": {...}, "reference": {...}}, ...], "top_k": int\|null}` | `{"results": [{"rank", "candidate_entity_id", "reference_entity_id", "score", "prediction"}], "count"}` |

### Request/response examples

```
POST /predict
{
  "candidate": {"entity_id": "s2_1", "business_name": "Acme Corp",
                 "business_address": "123 Main St 94105", "country": "USA"},
  "reference": {"entity_id": "s1_1", "business_name": "Acme Corporation",
                 "business_address": "123 Main Street 94105", "country": "USA"}
}
```
```
{
  "score": 0.97,
  "prediction": 1,
  "threshold": 0.11,
  "candidate_entity_id": "s2_1",
  "reference_entity_id": "s1_1"
}
```

## Error handling

- Missing/invalid input (bad payload shape, missing required fields, bad `top_k`) → HTTP 400
  with a specific, safe message (never a stack trace).
- Model not yet trained / not found → HTTP 503 (service not ready), not 500 — `/health` reports
  `"status": "degraded"` (still HTTP 200 for `/health` itself) rather than raising, since a health
  check must always answer.
- Any other unhandled exception → a generic `{"detail": "Internal server error"}` at HTTP 500 via
  a catch-all exception handler — internal exception details/tracebacks are never serialized into
  a response body, per spec.

## Testing strategy

`src/inference.py` and `src/ranking.py` depend only on packages already used in Phases 1-6
(numpy, joblib, scikit-learn) and are tested directly with this repo's existing
assert-based test convention (no `pytest` — matches every prior test file in this project, and
this sandbox has no `pytest` installed).

`src/api.py`'s request-handling functions (`handle_health`/`handle_predict`/`handle_rank`) are
plain functions with **no FastAPI/pydantic dependency**, so `tests/test_api.py` exercises them
directly regardless of whether FastAPI is installed. A second set of tests in the same file uses
`fastapi.testclient.TestClient` to hit the actual HTTP routes, but only if FastAPI is importable
in the environment — otherwise they print `SKIP` and pass trivially, rather than failing due to
an environment limitation unrelated to the code's correctness.

**This sandbox has no network access and could not install FastAPI/uvicorn/pydantic** — those
three routes were therefore written and reviewed carefully but not executed as a live HTTP
server in this environment. The install/run/verify commands below should be run on a machine
with internet access.

## Known limitations

- FastAPI/uvicorn/pydantic are new dependencies for this project (Phases 1-6 used only
  pandas/numpy/scikit-learn/joblib) — necessary because Phase 7 explicitly requires a REST API
  and prefers FastAPI; no other framework/ORM/etc. was added beyond this minimal set.
- The live HTTP endpoint tests could not be executed in this sandbox (no network to install
  FastAPI) — verify them on a machine with internet access using the commands below.
- The model is loaded once per process and cached (`api.get_model()`); a newly retrained model
  requires restarting the API process — there is no hot-reload endpoint (out of scope: no
  deployment/serving infrastructure was requested).
- `predict_batch`/`/rank` score each pair with an independent call to `predict_proba` on a
  single-row array; for very large batches a single batched `predict_proba` call would be more
  efficient, but this keeps `predict_pair` as the single source of truth for scoring logic
  (simplicity over micro-optimization, consistent with this still being an inference layer, not
  a high-throughput serving system).
- No authentication/rate-limiting/observability was added — explicitly out of scope
  ("do not start deployment/cloud infrastructure").

## Running locally

```
pip install fastapi uvicorn pydantic
python src/train_model_v2.py        # produces models/phase5_model.joblib + metadata
python src/phase6_pipeline.py       # produces models/phase6_threshold.json
uvicorn src.api:app --reload
```
Then:
```
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "candidate": {"entity_id": "s2_1", "business_name": "Acme Corp",
                 "business_address": "123 Main St 94105", "country": "USA"},
  "reference": {"entity_id": "s1_1", "business_name": "Acme Corporation",
                 "business_address": "123 Main Street 94105", "country": "USA"}
}'

curl -X POST http://127.0.0.1:8000/rank -H "Content-Type: application/json" -d '{
  "pairs": [
    {"candidate": {"entity_id": "s2_1", "business_name": "Acme Corp",
                    "business_address": "123 Main St 94105", "country": "USA"},
     "reference": {"entity_id": "s1_1", "business_name": "Acme Corporation",
                    "business_address": "123 Main Street 94105", "country": "USA"}}
  ],
  "top_k": 5
}'
```

Run the tests:
```
python tests/test_normalization.py           # Phase 2, unchanged, still passing
python tests/test_blocking.py                # Phase 2, unchanged, still passing
python tests/test_features.py                # Phase 3, unchanged, still passing
python tests/test_build_feature_dataset.py    # Phase 3, unchanged, still passing
python tests/test_baseline.py                 # Phase 4, unchanged, still passing
python tests/test_model_v2.py                 # Phase 5, unchanged, still passing
python tests/test_phase6.py                   # Phase 6, unchanged, still passing
python tests/test_inference.py                # Phase 7, new
python tests/test_ranking.py                  # Phase 7, new
python tests/test_api.py                      # Phase 7, new
```
