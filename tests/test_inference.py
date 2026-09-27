import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import FEATURE_NAMES  # noqa: E402
from inference import (  # noqa: E402
    InferenceModel,
    InvalidRecordError,
    ModelNotFoundError,
    PredictionResult,
    SchemaMismatchError,
    load_model,
)

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


def test_load_model_from_repo_models_dir():
    # Assumes `python src/train_model_v2.py` (and phase6_pipeline.py) have
    # already been run in this repo, producing models/*.
    model = load_model()
    assert isinstance(model, InferenceModel)
    assert model.feature_columns == list(FEATURE_NAMES)
    assert isinstance(model.threshold, float)


def test_load_model_missing_raises_clear_error(tmp_path):
    _expect_raises(ModelNotFoundError, load_model, model_dir=tmp_path)  # empty dir, no model files


def test_predict_pair_output_shape():
    model = load_model()
    result = model.predict_pair(GOOD_CANDIDATE, GOOD_REFERENCE)
    assert isinstance(result, PredictionResult)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.prediction in (0, 1)
    assert result.threshold == model.threshold
    assert result.candidate_entity_id == "s2_1"
    assert result.reference_entity_id == "s1_1"


def test_predict_pair_deterministic():
    model = load_model()
    r1 = model.predict_pair(GOOD_CANDIDATE, GOOD_REFERENCE)
    r2 = model.predict_pair(GOOD_CANDIDATE, GOOD_REFERENCE)
    assert r1.score == r2.score
    assert r1.prediction == r2.prediction


def test_threshold_behavior_matches_score_vs_threshold():
    model = load_model()
    result = model.predict_pair(GOOD_CANDIDATE, GOOD_REFERENCE)
    expected_prediction = int(result.score >= model.threshold)
    assert result.prediction == expected_prediction


def test_predict_pair_missing_required_field_raises():
    model = load_model()
    bad_candidate = {"entity_id": "s2_1", "business_name": "Acme Corp"}  # missing address/country
    _expect_raises(InvalidRecordError, model.predict_pair, bad_candidate, GOOD_REFERENCE)


def test_predict_pair_non_dict_record_raises():
    model = load_model()
    _expect_raises(InvalidRecordError, model.predict_pair, "not a dict", GOOD_REFERENCE)


def test_predict_pair_empty_field_values_do_not_crash():
    # Empty VALUES (as opposed to missing KEYS) are valid input -- must not
    # raise, since normalize_* handles empty strings gracefully already.
    model = load_model()
    empty_candidate = {"entity_id": "c", "business_name": "", "business_address": "", "country": ""}
    result = model.predict_pair(empty_candidate, GOOD_REFERENCE)
    assert isinstance(result.score, float)


def test_predict_batch_shape_and_order():
    model = load_model()
    pairs = [(GOOD_CANDIDATE, GOOD_REFERENCE), (GOOD_CANDIDATE, GOOD_REFERENCE)]
    results = model.predict_batch(pairs)
    assert len(results) == 2
    assert all(isinstance(r, PredictionResult) for r in results)


def test_predict_batch_fails_clearly_on_bad_pair():
    model = load_model()
    pairs = [(GOOD_CANDIDATE, GOOD_REFERENCE), ({"business_name": "X"}, GOOD_REFERENCE)]
    _expect_raises(InvalidRecordError, model.predict_batch, pairs)


def test_schema_mismatch_detected():
    model = load_model()
    _expect_raises(
        SchemaMismatchError, InferenceModel,
        pipeline=model._pipeline, feature_columns=["some_other_feature"], threshold=0.5, metadata={},
    )


def test_no_training_happens_during_inference():
    # predict_pair/predict_batch must never call .fit -- verified by
    # confirming the pipeline's classifier is already fitted (has learned
    # attributes) both before and after a prediction, with no error raised
    # by sklearn's "not fitted" checks either time.
    from sklearn.utils.validation import check_is_fitted

    model = load_model()
    check_is_fitted(model._pipeline.named_steps["clf"])
    model.predict_pair(GOOD_CANDIDATE, GOOD_REFERENCE)
    check_is_fitted(model._pipeline.named_steps["clf"])  # still fitted, unchanged


if __name__ == "__main__":
    import inspect
    import tempfile

    test_fns = [obj for name, obj in list(globals().items())
                if name.startswith("test_") and inspect.isfunction(obj)]
    failures = 0
    for fn in test_fns:
        try:
            sig = inspect.signature(fn)
            if "tmp_path" in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
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
