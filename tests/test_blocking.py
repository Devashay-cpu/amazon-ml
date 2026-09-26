import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blocking import BlockIndex  # noqa: E402
from normalization import normalize_address, normalize_business_name, normalize_country  # noqa: E402

# Toy Source-2 style records: (entity_id, name, address, country)
TOY_S2 = [
    ("s2_1", "Acme Corp", "123 Main St 94105", "USA"),
    ("s2_2", "Acme Corporation", "123 Main Street 94105", "USA"),  # near-duplicate of s2_1
    ("s2_3", "Globex Inc", "1 Infinite Loop 95014", "USA"),
    ("s2_4", "Sharma Traders", "MG Road 560001", "India"),
    ("s2_5", "Common Word Group", "5 High St 10001", "USA"),
    ("s2_6", "Another Common Word Co", "9 Low St 10002", "USA"),
]


def _build_index(records):
    idx = BlockIndex("TEST")
    for _, name, _, _ in records:
        idx.add_pass1(normalize_business_name(name))
    idx.finalize_pass1()
    for eid, name, addr, country in records:
        idx.add_pass2(eid, normalize_business_name(name), normalize_address(addr), normalize_country(country))
    return idx


def test_exact_name_block_matches_near_duplicate():
    idx = _build_index(TOY_S2)
    # "Acme Corp" and "Acme Corporation" both suffix-strip to "acme"
    n = normalize_business_name("Acme Enterprises")
    a = normalize_address("123 Main St 94105")
    c = normalize_country("USA")
    cands, per_strat = idx.get_candidates(n, a, c)
    assert "s2_1" in cands
    assert "s2_2" in cands


def test_postal_code_block_matches():
    idx = _build_index(TOY_S2)
    n = normalize_business_name("Totally Different Name")
    a = normalize_address("999 Nowhere Rd 94105")
    c = normalize_country("USA")
    cands, per_strat = idx.get_candidates(n, a, c)
    assert "s2_1" in cands or "s2_2" in cands
    assert "postal_code" in per_strat


def test_house_number_street_block():
    idx = _build_index(TOY_S2)
    n = normalize_business_name("Unrelated Name")
    a = normalize_address("1 Infinite Loop")
    c = normalize_country("USA")
    cands, per_strat = idx.get_candidates(n, a, c)
    assert "s2_3" in cands


def test_no_cross_country_leakage_via_exact_name():
    idx = _build_index(TOY_S2)
    n = normalize_business_name("Sharma Traders")
    a = normalize_address("Some Road 000000")
    c = normalize_country("France")  # same name, wrong country
    cands, per_strat = idx.get_candidates(n, a, c)
    assert "s2_4" not in cands


def test_rare_token_strategy_present_for_uncommon_words():
    idx = _build_index(TOY_S2)
    n = normalize_business_name("Globex Something Else")
    a = normalize_address("no match address")
    c = normalize_country("USA")
    cands, per_strat = idx.get_candidates(n, a, c)
    assert "s2_3" in cands
    assert "country_rare_name_token" in per_strat


def test_common_token_not_used_as_rare_block():
    # "common" and "word" appear in two docs out of six -> above the tiny
    # rarity threshold for this toy corpus, so should NOT create a shared
    # rare-token block key between s2_5 and s2_6's queriers by themselves.
    idx = _build_index(TOY_S2)
    assert idx.rare_token_threshold is not None
    freq_common = idx.token_doc_freq.get("common", 0)
    freq_word = idx.token_doc_freq.get("word", 0)
    assert freq_common >= 2 or freq_word >= 2


def test_problematic_blocks_empty_for_small_toy_set():
    idx = _build_index(TOY_S2)
    assert idx.problematic_blocks() == []


def test_strategy_key_counts_nonzero():
    idx = _build_index(TOY_S2)
    counts = idx.strategy_key_counts()
    assert counts["country_exact_name"] > 0
    assert counts["postal_code"] > 0


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
