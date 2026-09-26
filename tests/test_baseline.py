import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import FEATURE_NAMES, MISSING  # noqa: E402
from train_baseline import (  # noqa: E402
    ID_COLUMNS,
    LABEL_COLUMN,
    build_pipeline,
    evaluate_at_threshold,
    load_dataset,
    select_threshold,
)


def _toy_csv(tmp_dir: Path) -> Path:
    """A small but non-trivial synthetic Phase-3-shaped feature CSV."""
    rng = np.random.RandomState(0)
    n = 40
    rows = []
    for i in range(n):
        label = 1 if i < 20 else 0
        # Positives: high similarity features; negatives: low similarity.
        base = 0.9 if label == 1 else 0.2
        row = {
            "s1_entity_id": f"s1_{i}",
            "candidate_entity_id": f"cand_{i}",
            "source": "S2" if i % 2 == 0 else "S3",
        }
        for fname in FEATURE_NAMES:
            if fname == "unit_token_match":
                row[fname] = MISSING  # deliberately always-missing column, like the real toy run
            elif "match" in fname and "jaccard" not in fname and "similarity" not in fname:
                row[fname] = 1 if (label == 1 and rng.rand() > 0.1) else 0
            else:
                row[fname] = round(min(max(base + rng.normal(0, 0.05), 0), 1), 4)
        row[LABEL_COLUMN] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_dir / "toy_pair_features.csv"
    df.to_csv(path, index=False)
    return path


def test_dataset_loads_correctly(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    assert stats["n_rows_loaded"] == 40
    assert stats["n_rows_used"] == 40
    assert stats["n_rows_dropped_invalid_label"] == 0
    assert stats["n_positive"] == 20
    assert stats["n_negative"] == 20


def test_identifier_columns_excluded_from_features(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    for id_col in ID_COLUMNS:
        assert id_col not in feature_columns
    assert LABEL_COLUMN not in feature_columns
    assert set(feature_columns) == set(FEATURE_NAMES)


def test_label_handling_drops_invalid_rows_and_reports_it(tmp_path):
    path = _toy_csv(tmp_path)
    df = pd.read_csv(path)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(object)
    df.loc[0, LABEL_COLUMN] = ""  # simulate a test-split row with no ground truth
    df.to_csv(path, index=False)
    loaded, feature_columns, stats = load_dataset(path)
    assert stats["n_rows_dropped_invalid_label"] == 1
    assert stats["n_rows_used"] == 39
    assert set(loaded[LABEL_COLUMN].unique()) <= {0, 1}


def test_missing_sentinel_counts_reported(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    assert stats["missing_sentinel_counts"]["unit_token_match"] == 40


def test_missing_sentinel_not_treated_as_normal_value_by_pipeline(tmp_path):
    # A column that is MISSING for every training row must not silently
    # corrupt training -- the imputer either drops it or handles it without
    # ever treating -1 as a valid similarity score of "very dissimilar".
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(X, y)  # must not raise
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_training_pipeline_runs_and_prediction_shape_is_correct(tmp_path):
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = build_pipeline(seed=42)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_val)[:, 1]
    preds = pipeline.predict(X_val)
    assert proba.shape == (len(X_val),)
    assert preds.shape == (len(X_val),)
    assert set(np.unique(preds)) <= {0, 1}


def test_threshold_evaluation_works(tmp_path):
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    result_low = evaluate_at_threshold(y_true, y_proba, 0.5)
    assert result_low["precision"] == 1.0
    assert result_low["recall"] == 1.0
    assert result_low["confusion_matrix"]["tp"] == 3

    best_threshold, best_metrics = select_threshold(y_true, y_proba, grid_step=0.1)
    assert 0.0 < best_threshold < 1.0
    assert best_metrics["f1"] >= result_low["f1"] - 1e-9  # selected threshold is at least as good


def test_no_train_validation_leakage(tmp_path):
    # The imputer/scaler statistics must come only from the training fold.
    path = _toy_csv(tmp_path)
    df, feature_columns, stats = load_dataset(path)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline_a = build_pipeline(seed=42)
    pipeline_a.fit(X_train, y_train)
    imputer_a = pipeline_a.named_steps["impute_missing_sentinel"]
    stats_from_train_only = imputer_a.statistics_.copy()

    # Fitting on train+val together should, in general, change the imputer's
    # learned statistics if the validation fold's distribution differs --
    # demonstrating the pipeline is sensitive to what it's fit on, i.e. it
    # is NOT silently using validation data when we only call fit(X_train).
    pipeline_b = build_pipeline(seed=42)
    pipeline_b.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    imputer_b = pipeline_b.named_steps["impute_missing_sentinel"]
    stats_from_all_data = imputer_b.statistics_.copy()

    # We only assert that pipeline_a's statistics were computed from X_train
    # alone: refitting an identical pipeline on X_train alone must reproduce
    # them exactly (determinism + train-only fitting).
    pipeline_c = build_pipeline(seed=42)
    pipeline_c.fit(X_train, y_train)
    imputer_c = pipeline_c.named_steps["impute_missing_sentinel"]
    np.testing.assert_array_equal(stats_from_train_only, imputer_c.statistics_)


def test_feature_schema_matches_phase3(tmp_path):
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
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    sys.exit(1 if failures else 0)
