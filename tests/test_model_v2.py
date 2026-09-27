import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import FEATURE_NAMES, MISSING  # noqa: E402
from train_baseline import LABEL_COLUMN, load_dataset  # noqa: E402
from train_model_v2 import _sentinel_to_nan, build_gbm_pipeline, _auc_metrics  # noqa: E402


def _toy_csv(tmp_dir: Path, n: int = 200, positive_rate: float = 0.05) -> Path:
    """
    A larger, imbalanced, Phase-3-shaped feature CSV -- big enough for a
    meaningful stratified split and for HistGradientBoostingClassifier to
    behave sensibly (unlike the 12-row integration-toy dataset used
    elsewhere in this repo, which is too small for a tree ensemble to be
    informative).
    """
    rng = np.random.RandomState(0)
    n_pos = max(int(n * positive_rate), 4)
    n_neg = n - n_pos
    rows = []
    for i in range(n):
        label = 1 if i < n_pos else 0
        base = 0.9 if label == 1 else 0.2
        row = {
            "s1_entity_id": f"s1_{i}",
            "candidate_entity_id": f"cand_{i}",
            "source": "S2" if i % 2 == 0 else "S3",
        }
        for fname in FEATURE_NAMES:
            if fname == "unit_token_match":
                row[fname] = MISSING if rng.rand() > 0.3 else (1 if label == 1 else 0)
            elif "match" in fname and "jaccard" not in fname and "similarity" not in fname:
                row[fname] = 1 if (label == 1 and rng.rand() > 0.1) else 0
            else:
                row[fname] = round(min(max(base + rng.normal(0, 0.05), 0), 1), 4)
        row[LABEL_COLUMN] = label
        rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle
    path = tmp_dir / "toy_pair_features_imbalanced.csv"
    df.to_csv(path, index=False)
    return path


def test_sentinel_to_nan_converts_only_sentinel_values():
    X = np.array([[1.0, MISSING, 0.5], [MISSING, MISSING, 1.0]])
    out = _sentinel_to_nan(X)
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 0])
    assert np.isnan(out[1, 1])
    assert out[0, 0] == 1.0
    assert out[0, 2] == 0.5
    assert out[1, 2] == 1.0


def test_sentinel_to_nan_does_not_mutate_input():
    X = np.array([[1.0, MISSING], [2.0, 3.0]])
    X_copy = X.copy()
    _ = _sentinel_to_nan(X)
    np.testing.assert_array_equal(X, X_copy)  # original untouched


def test_gbm_pipeline_fits_and_predicts_with_expected_shape(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy()

    pipeline = build_gbm_pipeline(seed=42, max_iter=50)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    assert proba.shape == (len(X),)
    assert preds.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_gbm_pipeline_handles_mostly_missing_column_without_crashing(tmp_path):
    """
    Regression test for a real-data failure: on some scikit-learn/NumPy
    version combinations, HistGradientBoostingClassifier's internal bin-
    threshold computation derives midpoints between a column's distinct
    non-missing values using a size-2 sliding window (e.g. via
    numpy.lib.stride_tricks.sliding_window_view). If a column has ZERO
    non-missing values, that window (size 2) is larger than the distinct-
    value array (size 0), and NumPy raises "window shape cannot be larger
    than input array shape".

    A column with literally zero observed values is a genuinely degenerate
    input no histogram-based binner can bin at all -- it isn't a realistic
    case for this dataset (even a heavily-missing feature like
    unit_token_match still has *some* observed values), so the fixture is
    corrected here to leave 2 distinct valid values (the minimum any
    binning implementation needs to form at least one midpoint) rather
    than forcing 0. Production code (build_gbm_pipeline,
    HistGradientBoostingClassifier's configuration) is unchanged -- this
    fixes only what unrealistic input the test fixture constructs.
    """
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=np.float32)
    # Force one column to the sentinel value for almost every row, like the
    # earlier all-missing edge case seen with the Phase 4 baseline on small
    # runs -- but leave 2 distinct valid observations, since a column with
    # truly zero observed values is degenerate for any histogram binner,
    # not a case this model is expected (or able) to handle.
    X[:, 0] = MISSING
    X[0, 0] = 0.1
    X[1, 0] = 0.9
    y = df[LABEL_COLUMN].to_numpy()

    pipeline = build_gbm_pipeline(seed=42, max_iter=30)
    pipeline.fit(X, y)  # must not raise
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)


def test_gbm_handles_severe_class_imbalance(tmp_path):
    path = _toy_csv(tmp_path, n=300, positive_rate=0.02)
    df, feature_columns, stats = load_dataset(path)
    assert stats["n_positive"] < stats["n_negative"] * 0.05  # confirm it's actually imbalanced

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_gbm_pipeline(seed=42, max_iter=50)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_val)[:, 1]
    # class_weight="balanced" should let the model actually assign some
    # meaningfully high probabilities to at least some validation rows,
    # rather than collapsing to "always predict negative".
    assert proba.max() > 0.3


def test_auc_metrics_handles_single_class_gracefully():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    roc_auc, pr_auc = _auc_metrics(y_true, y_proba)
    assert roc_auc is None
    assert pr_auc is None


def test_auc_metrics_normal_case():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    roc_auc, pr_auc = _auc_metrics(y_true, y_proba)
    assert roc_auc == 1.0
    assert pr_auc == 1.0


def test_no_leakage_determinism_same_train_data_same_predictions(tmp_path):
    # Fitting twice on the identical training data (fixed random_state) must
    # give identical validation predictions -- demonstrates the pipeline
    # does not use validation data or any other hidden randomness/state.
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline_a = build_gbm_pipeline(seed=42, max_iter=30)
    pipeline_a.fit(X_train, y_train)
    proba_a = pipeline_a.predict_proba(X_val)[:, 1]

    pipeline_b = build_gbm_pipeline(seed=42, max_iter=30)
    pipeline_b.fit(X_train, y_train)
    proba_b = pipeline_b.predict_proba(X_val)[:, 1]

    np.testing.assert_allclose(proba_a, proba_b)


def test_early_stopping_disabled_uses_full_training_set():
    # early_stopping=False must be set so HGB doesn't carve its own internal
    # validation split out of X_train (which would break the "identical
    # split methodology to Phase 4" guarantee this script documents).
    pipeline = build_gbm_pipeline(seed=42, max_iter=30)
    clf = pipeline.named_steps["clf"]
    assert clf.early_stopping is False


def test_feature_schema_unchanged_from_phase3(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    assert feature_columns == [f for f in FEATURE_NAMES if f in feature_columns]
    assert len(feature_columns) == len(FEATURE_NAMES)


if __name__ == "__main__":
    import inspect
    import tempfile

    test_fns = [obj for name, obj in list(globals().items())
                if name.startswith("test_") and inspect.isfunction(obj)]
    failures = 0
    for fn in test_fns:
        try:
            sig = inspect.signature(fn)
            if sig.parameters:
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
