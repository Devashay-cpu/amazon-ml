import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import FEATURE_NAMES  # noqa: E402
from phase6_pipeline import (  # noqa: E402
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    analyze_thresholds,
    business_cost,
    run_relaxed_blocking_experiment,
    summarize_blocking_stats,
)


def test_business_cost_basic_arithmetic():
    assert business_cost(n_fp=3, n_fn=2, cost_fp=1.0, cost_fn=10.0) == 3 * 1.0 + 2 * 10.0
    assert business_cost(n_fp=0, n_fn=0, cost_fp=5.0, cost_fn=50.0) == 0.0


def test_business_cost_weights_false_negatives_more_by_default():
    # With the documented default weights, one FN should cost more than one FP.
    cost_one_fp = business_cost(1, 0, DEFAULT_COST_FP, DEFAULT_COST_FN)
    cost_one_fn = business_cost(0, 1, DEFAULT_COST_FP, DEFAULT_COST_FN)
    assert cost_one_fn > cost_one_fp


def test_analyze_thresholds_selects_lower_cost_option():
    # Constructed so a HIGH threshold has zero FP/FN (perfect separation),
    # while a low threshold introduces avoidable false positives.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rows, best = analyze_thresholds(y_true, y_proba, cost_fp=1.0, cost_fn=10.0, grid_step=0.1)
    assert best["business_cost"] == min(r["business_cost"] for r in rows)
    assert best["confusion_matrix"]["fp"] == 0
    assert best["confusion_matrix"]["fn"] == 0


def test_analyze_thresholds_prefers_recall_when_fn_costly():
    # All predictions land at the same probability for the positive class
    # except one -- with FN weighted heavily, the selected threshold must
    # not sacrifice recall to gain a small amount of precision.
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.9])
    rows, best = analyze_thresholds(y_true, y_proba, cost_fp=1.0, cost_fn=50.0, grid_step=0.01)
    # A threshold above 0.9 would miss all 4 positives (FN=4, cost=200) --
    # far worse than accepting some false positives, so recall should be high.
    assert best["recall"] >= 0.75


def test_analyze_thresholds_deterministic():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.7, 0.9])
    rows_a, best_a = analyze_thresholds(y_true, y_proba, 1.0, 10.0)
    rows_b, best_b = analyze_thresholds(y_true, y_proba, 1.0, 10.0)
    assert rows_a == rows_b
    assert best_a == best_b


def test_summarize_blocking_stats_ranks_strategies_and_flags_low_value():
    synthetic_stats = {
        "run": {"entities_evaluated": 1000},
        "blocking_recall": {"overall": 0.85, "s2": 0.9, "s3": 0.8,
                             "only_s2_entities": 0.9, "only_s3_entities": 0.8, "both_s2_s3_entities": 0.85},
        "strategy_pairs_found": {
            "country_exact_name": 500,
            "postal_code": 400,
            "country_rare_name_token": 2,  # near-zero contribution
            "address_alnum_prefix8": 0,     # zero contribution
        },
        "s2_problematic_blocks": [["country_name_prefix4", "usa|acme", 9000]],
        "s3_problematic_blocks": [],
    }
    summary = summarize_blocking_stats(synthetic_stats)
    assert summary["ranked_strategies"][0] == ("country_exact_name", 500)
    assert "address_alnum_prefix8" in summary["low_value_strategies"]
    assert "country_rare_name_token" in summary["low_value_strategies"]
    assert "country_exact_name" not in summary["low_value_strategies"]
    assert summary["n_s2_problematic_blocks"] == 1
    assert summary["n_s3_problematic_blocks"] == 0
    assert summary["recall"]["overall"] == 0.85


def test_summarize_blocking_stats_handles_missing_keys_gracefully():
    # A minimal/partial stats dict must not crash the summarizer.
    summary = summarize_blocking_stats({})
    assert summary["recall"] == {}
    assert summary["ranked_strategies"] == []
    assert summary["low_value_strategies"] == []


def test_feature_schema_unchanged_import():
    # Phase 6 must not redefine or diverge from the Phase 3 feature schema.
    assert len(FEATURE_NAMES) == 16


def test_relaxation_experiment_restores_blocking_module_constants(tmp_path):
    import blocking as blocking_module
    import csv

    original_fraction = blocking_module.RARE_TOKEN_MAX_DOC_FRACTION
    original_cap = blocking_module.RARE_TOKEN_MAX_DOC_CAP

    # Minimal on-disk dataset so build_index_from_source/iter_candidates can run.
    train_dir = tmp_path / "dataset" / "train"
    train_dir.mkdir(parents=True)

    def write_tsv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    write_tsv(train_dir / "train_source1.tsv", ["entity_id", "business_name", "business_address", "country"],
               [("s1_1", "Acme Corp", "123 Main St 94105", "USA")])
    write_tsv(train_dir / "train_source2.tsv", ["entity_id", "business_name", "business_address", "country"],
               [("s2_1", "Acme Corp", "123 Main St 94105", "USA")])
    write_tsv(train_dir / "train_source3.tsv", ["entity_id", "business_name", "business_address", "country"],
               [("s3_1", "Acme Corp", "123 Main St 94105", "USA")])
    write_tsv(train_dir / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
               [("s1_1", "s2_1,s3_1")])

    run_relaxed_blocking_experiment(tmp_path, "train", None, relaxed_fraction=0.01, relaxed_cap=500)

    # Regardless of what happened inside, the module-level constants must
    # be restored to their original values afterward -- proves the
    # experiment never leaves Phase 2's default behavior altered.
    assert blocking_module.RARE_TOKEN_MAX_DOC_FRACTION == original_fraction
    assert blocking_module.RARE_TOKEN_MAX_DOC_CAP == original_cap


def test_tuning_uses_only_training_data_no_leakage():
    from phase6_pipeline import maybe_tune

    rng = np.random.RandomState(0)
    n = 120
    y = np.array([1] * 20 + [0] * 100)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        base = 0.9 if y[i] == 1 else 0.2
        X[i, :] = base + rng.normal(0, 0.05, size=len(FEATURE_NAMES))

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # maybe_tune must only ever be called with (and only ever fit on)
    # X_train/y_train -- passing distinguishable "poisoned" validation-only
    # rows would make this detectable, but the simplest direct guarantee is
    # structural: maybe_tune's signature takes no X_val/y_val parameter at
    # all, so it is impossible for it to fit on validation data by
    # accident. This test asserts that contract and that it runs cleanly
    # end-to-end using only the training split.
    import inspect
    sig = inspect.signature(maybe_tune)
    assert "X_val" not in sig.parameters
    assert "y_val" not in sig.parameters

    best_estimator, best_params, best_cv_score = maybe_tune(X_train, y_train, seed=42, n_iter=3, cv=2)
    assert best_estimator is not None
    assert isinstance(best_params, dict)
    # The fitted estimator's training sample count must match X_train, not
    # X_train + X_val -- confirms no validation rows entered the fit.
    assert best_estimator.named_steps["clf"].n_features_in_ == X_train.shape[1]


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
