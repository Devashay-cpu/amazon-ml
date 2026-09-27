import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import api  # noqa: E402

GOOD_CANDIDATE = {"entity_id": "s2_1", "business_name": "Acme Corp",
                  "business_address": "123 Main St 94105", "country": "USA"}
GOOD_REFERENCE = {"entity_id": "s1_1", "business_name": "Acme Corporation",
                  "business_address": "123 Main Street 94105", "country": "USA"}


def _expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__} to be raised, nothing was")


# --------------------------------------------------------------------------
# Framework-agnostic handler tests -- always run, no FastAPI/pydantic needed.
# Assumes `python src/train_model_v2.py` + `python src/phase6_pipeline.py`
# have already been run in this repo (models/ populated).
# --------------------------------------------------------------------------

def test_health_reports_model_loaded():
    api.reset_model_cache()
    result = api.handle_health()
    assert result["status"] == "ok"
    assert result["model_loaded"] is True
    assert result["feature_count"] == 16
    assert isinstance(result["threshold"], float)


def test_health_never_raises_even_if_model_dir_missing(monkeypatch=None):
    # Simulate a missing model by pointing get_model's cache lookup at a
    # broken loader, without touching the real models/ directory.
    api.reset_model_cache()
    original_load_model = api.load_model

    def broken_load_model(*args, **kwargs):
        raise api.ModelNotFoundError("simulated: model not found")

    api.load_model = broken_load_model
    try:
        result = api.handle_health()
        assert result["status"] == "degraded"
        assert result["model_loaded"] is False
        assert "detail" in result
    finally:
        api.load_model = original_load_model
        api.reset_model_cache()


def test_predict_handler_happy_path():
    api.reset_model_cache()
    result = api.handle_predict({"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE})
    assert set(result.keys()) == {"score", "prediction", "threshold", "candidate_entity_id", "reference_entity_id"}
    assert 0.0 <= result["score"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["candidate_entity_id"] == "s2_1"


def test_predict_handler_rejects_non_dict_payload():
    _expect_raises(api.ApiError, api.handle_predict, "not a dict")


def test_predict_handler_rejects_missing_candidate_or_reference():
    _expect_raises(api.ApiError, api.handle_predict, {"candidate": GOOD_CANDIDATE})
    _expect_raises(api.ApiError, api.handle_predict, {"reference": GOOD_REFERENCE})
    _expect_raises(api.ApiError, api.handle_predict, {})


def test_predict_handler_rejects_invalid_record_with_400():
    try:
        api.handle_predict({"candidate": {"business_name": "X"}, "reference": GOOD_REFERENCE})
        raise AssertionError("expected ApiError")
    except api.ApiError as e:
        assert e.status_code == 400


def test_rank_handler_happy_path_sorted_descending():
    api.reset_model_cache()
    pairs = [
        {"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE},
        {"candidate": {"entity_id": "s2_9", "business_name": "Zephyr Dynamics",
                       "business_address": "9 Nowhere Ave 00000", "country": "France"},
         "reference": GOOD_REFERENCE},
    ]
    result = api.handle_rank({"pairs": pairs})
    assert result["count"] == 2
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in result["results"]] == [1, 2]


def test_rank_handler_top_k():
    api.reset_model_cache()
    pairs = [{"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE} for _ in range(5)]
    result = api.handle_rank({"pairs": pairs, "top_k": 2})
    assert result["count"] == 2


def test_rank_handler_rejects_empty_pairs():
    _expect_raises(api.ApiError, api.handle_rank, {"pairs": []})
    _expect_raises(api.ApiError, api.handle_rank, {})


def test_rank_handler_rejects_bad_top_k():
    _expect_raises(api.ApiError, api.handle_rank, {"pairs": [{"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE}], "top_k": -1})
    _expect_raises(api.ApiError, api.handle_rank, {"pairs": [{"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE}], "top_k": "two"})


def test_rank_handler_rejects_pair_missing_candidate_or_reference():
    _expect_raises(api.ApiError, api.handle_rank, {"pairs": [{"candidate": GOOD_CANDIDATE}]})


def test_rank_handler_propagates_invalid_record_as_400():
    try:
        api.handle_rank({"pairs": [{"candidate": {"business_name": "X"}, "reference": GOOD_REFERENCE}]})
        raise AssertionError("expected ApiError")
    except api.ApiError as e:
        assert e.status_code == 400


def test_model_is_cached_across_calls_not_reloaded_every_request():
    api.reset_model_cache()
    m1 = api.get_model()
    m2 = api.get_model()
    assert m1 is m2  # same object -- confirms no reloading/retraining per call


# --------------------------------------------------------------------------
# Optional: live FastAPI endpoint tests, only if fastapi (+ its test client
# dependencies) are actually installed. Gracefully skipped otherwise, since
# this environment may not have network access to install them -- the
# handler-level tests above already cover the same request/response logic.
# --------------------------------------------------------------------------

def _fastapi_testclient_available():
    if api.app is None:
        return False
    try:
        from fastapi.testclient import TestClient  # noqa: F401
        return True
    except ImportError:
        return False


def test_live_health_endpoint_if_fastapi_available():
    if not _fastapi_testclient_available():
        print("SKIP (fastapi/testclient not installed in this environment)")
        return
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_live_predict_endpoint_if_fastapi_available():
    if not _fastapi_testclient_available():
        print("SKIP (fastapi/testclient not installed in this environment)")
        return
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    resp = client.post("/predict", json={"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE})
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body and "prediction" in body and "threshold" in body


def test_live_rank_endpoint_if_fastapi_available():
    if not _fastapi_testclient_available():
        print("SKIP (fastapi/testclient not installed in this environment)")
        return
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    resp = client.post("/rank", json={"pairs": [{"candidate": GOOD_CANDIDATE, "reference": GOOD_REFERENCE}]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_live_invalid_payload_returns_400_if_fastapi_available():
    if not _fastapi_testclient_available():
        print("SKIP (fastapi/testclient not installed in this environment)")
        return
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    resp = client.post("/predict", json={"candidate": {"business_name": "X"}, "reference": GOOD_REFERENCE})
    assert resp.status_code == 400
    assert "internal" not in resp.text.lower() or "traceback" not in resp.text.lower()


if __name__ == "__main__":
    import inspect

    test_fns = [obj for name, obj in list(globals().items())
                if name.startswith("test_") and inspect.isfunction(obj)]
    failures = 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    sys.exit(1 if failures else 0)
