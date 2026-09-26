import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import FEATURE_NAMES, MISSING, extract_pair_features  # noqa: E402
from normalization import normalize_address, normalize_business_name, normalize_country  # noqa: E402


def _record(name, address, country):
    return {
        "name": normalize_business_name(name),
        "address": normalize_address(address),
        "country": normalize_country(country),
    }


def test_identical_records_give_exact_matches():
    r = _record("Acme Corp", "123 Main St 94105", "USA")
    feats = extract_pair_features(r, r)
    assert feats["name_exact_match"] == 1
    assert feats["address_exact_match"] == 1
    assert feats["country_match"] == 1
    assert feats["name_char_similarity"] == 1.0
    assert feats["address_char_similarity"] == 1.0
    assert feats["name_token_jaccard"] == 1.0
    assert feats["house_number_match"] == 1
    assert feats["postal_code_match"] == 1


def test_clearly_different_records_give_low_similarity():
    r1 = _record("Acme Corp", "123 Main St 94105", "USA")
    r2 = _record("Zephyr Dynamics", "9 Nowhere Ave 00000", "France")
    feats = extract_pair_features(r1, r2)
    assert feats["name_exact_match"] == 0
    assert feats["address_exact_match"] == 0
    assert feats["country_match"] == 0
    assert feats["name_char_similarity"] < 0.5
    assert feats["name_token_jaccard"] in (0.0, MISSING)


def test_near_duplicate_name_word_order_and_legal_suffix():
    r1 = _record("Acme Corp", "1 Loop", "USA")
    r2 = _record("Acme Corporation", "1 Loop", "USA")
    feats = extract_pair_features(r1, r2)
    assert feats["name_suffix_stripped_match"] == 1  # both strip to "acme"
    assert feats["name_exact_match"] == 0            # "acme corp" != "acme corporation"


def test_missing_fields_do_not_crash_and_use_sentinel():
    r1 = _record(None, None, None)
    r2 = _record("Acme Corp", "123 Main St 94105", "USA")
    feats = extract_pair_features(r1, r2)  # must not raise
    assert feats["name_exact_match"] == MISSING
    assert feats["address_exact_match"] == MISSING
    assert feats["country_match"] == MISSING
    assert feats["house_number_match"] == MISSING
    assert feats["postal_code_match"] == MISSING


def test_both_missing_is_not_reported_as_a_confident_match():
    r1 = _record("", "", "")
    r2 = _record("", "", "")
    feats = extract_pair_features(r1, r2)
    # Two blanks must never look like a strong match (that would be a
    # data-quality / leakage risk — "no data" != "confirmed match").
    assert feats["name_exact_match"] == MISSING
    assert feats["name_char_similarity"] == MISSING
    assert feats["address_exact_match"] == MISSING


def test_feature_output_schema_is_stable_and_complete():
    r = _record("Acme Corp", "123 Main St 94105", "USA")
    feats = extract_pair_features(r, r)
    assert list(feats.keys()) == FEATURE_NAMES


def test_deterministic_output():
    r1 = _record("Acme Corp", "123 Main St 94105", "USA")
    r2 = _record("Acme Corporation", "123 Main Street 94105", "USA")
    feats_a = extract_pair_features(r1, r2)
    feats_b = extract_pair_features(r1, r2)
    assert feats_a == feats_b


def test_house_number_mismatch_detected():
    r1 = _record("Acme Corp", "123 Main St", "USA")
    r2 = _record("Acme Corp", "456 Main St", "USA")
    feats = extract_pair_features(r1, r2)
    assert feats["house_number_match"] == 0


def test_postal_code_mismatch_detected():
    r1 = _record("Acme Corp", "123 Main St 94105", "USA")
    r2 = _record("Acme Corp", "123 Main St 10001", "USA")
    feats = extract_pair_features(r1, r2)
    assert feats["postal_code_match"] == 0


# --- label-assignment + duplicate-pair logic, mirroring what
# build_feature_dataset.py does, without needing real files on disk ---

def test_label_assignment_logic():
    ground_truth = {"s1_1": {"s2": {"s2_1", "s2_2"}, "s3": {"s3_1"}}}
    gt = ground_truth["s1_1"]

    def label_for(source, cand_id):
        return 1 if cand_id in gt[source.lower()] else 0

    assert label_for("S2", "s2_1") == 1
    assert label_for("S2", "s2_9") == 0
    assert label_for("S3", "s3_1") == 1
    assert label_for("S3", "s3_9") == 0


def test_duplicate_candidate_pairs_are_deduplicated():
    seen_pairs = set()
    duplicate_count = 0
    incoming = [("s1_1", "S2", "s2_1"), ("s1_1", "S2", "s2_1"), ("s1_1", "S2", "s2_2")]
    rows_written = 0
    for pair_key in incoming:
        if pair_key in seen_pairs:
            duplicate_count += 1
            continue
        seen_pairs.add(pair_key)
        rows_written += 1
    assert rows_written == 2
    assert duplicate_count == 1


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
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    sys.exit(1 if failures else 0)
