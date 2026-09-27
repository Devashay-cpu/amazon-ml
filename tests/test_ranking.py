import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranking import InvalidScoredPairError, RankedResult, rank_candidates  # noqa: E402


def _expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__} to be raised, nothing was")


def test_ranks_descending_by_score():
    pairs = [
        {"candidate_entity_id": "c1", "score": 0.2, "prediction": 0},
        {"candidate_entity_id": "c2", "score": 0.9, "prediction": 1},
        {"candidate_entity_id": "c3", "score": 0.5, "prediction": 0},
    ]
    ranked = rank_candidates(pairs)
    assert [r.candidate_entity_id for r in ranked] == ["c2", "c3", "c1"]
    assert [r.score for r in ranked] == [0.9, 0.5, 0.2]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_top_k_truncates_after_full_sort():
    pairs = [
        {"candidate_entity_id": "c1", "score": 0.2},
        {"candidate_entity_id": "c2", "score": 0.9},
        {"candidate_entity_id": "c3", "score": 0.5},
        {"candidate_entity_id": "c4", "score": 0.7},
    ]
    ranked = rank_candidates(pairs, top_k=2)
    assert len(ranked) == 2
    assert [r.candidate_entity_id for r in ranked] == ["c2", "c4"]
    assert [r.rank for r in ranked] == [1, 2]


def test_top_k_zero_returns_empty():
    pairs = [{"candidate_entity_id": "c1", "score": 0.5}]
    ranked = rank_candidates(pairs, top_k=0)
    assert ranked == []


def test_top_k_larger_than_input_returns_all():
    pairs = [{"candidate_entity_id": "c1", "score": 0.5}, {"candidate_entity_id": "c2", "score": 0.9}]
    ranked = rank_candidates(pairs, top_k=100)
    assert len(ranked) == 2


def test_tie_handling_is_deterministic_by_candidate_id():
    pairs = [
        {"candidate_entity_id": "c3", "score": 0.5},
        {"candidate_entity_id": "c1", "score": 0.5},
        {"candidate_entity_id": "c2", "score": 0.5},
    ]
    ranked_a = rank_candidates(pairs)
    ranked_b = rank_candidates(list(reversed(pairs)))  # different input order, same tie set
    order_a = [r.candidate_entity_id for r in ranked_a]
    order_b = [r.candidate_entity_id for r in ranked_b]
    assert order_a == order_b == ["c1", "c2", "c3"]


def test_preserves_reference_and_candidate_ids():
    pairs = [{"candidate_entity_id": "c1", "reference_entity_id": "r1", "score": 0.5, "prediction": 1}]
    ranked = rank_candidates(pairs)
    assert ranked[0].candidate_entity_id == "c1"
    assert ranked[0].reference_entity_id == "r1"
    assert ranked[0].prediction == 1


def test_missing_score_raises():
    _expect_raises(InvalidScoredPairError, rank_candidates, [{"candidate_entity_id": "c1"}])


def test_non_numeric_score_raises():
    _expect_raises(InvalidScoredPairError, rank_candidates, [{"score": "not-a-number"}])


def test_non_list_input_raises():
    _expect_raises(InvalidScoredPairError, rank_candidates, "not a list")


def test_non_dict_element_raises():
    _expect_raises(InvalidScoredPairError, rank_candidates, [{"score": 0.5}, "not a dict"])


def test_negative_top_k_raises():
    _expect_raises(InvalidScoredPairError, rank_candidates, [{"score": 0.5}], top_k=-1)


def test_empty_list_returns_empty():
    assert rank_candidates([]) == []


def test_result_is_ranked_result_instance():
    ranked = rank_candidates([{"score": 0.5}])
    assert isinstance(ranked[0], RankedResult)
    d = ranked[0].to_dict()
    assert d["rank"] == 1
    assert d["score"] == 0.5


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
