import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from normalization import (  # noqa: E402
    normalize_address,
    normalize_business_name,
    normalize_country,
)


def test_name_lowercase_and_punctuation():
    n = normalize_business_name("Acme, Corp.")
    assert n["exact_normalized"] == "acme corp"


def test_name_suffix_stripping():
    n = normalize_business_name("Sharma & Sons Pvt. Ltd.")
    assert n["suffix_stripped"] == "sharma and sons"


def test_name_ampersand_expansion():
    n = normalize_business_name("A & B")
    assert "and" in n["exact_normalized"].split()


def test_name_alnum_has_no_spaces_or_punct():
    n = normalize_business_name("A-B & C, Inc.")
    assert n["alnum_normalized"].isalnum()
    assert " " not in n["alnum_normalized"]


def test_name_token_sorted():
    n = normalize_business_name("Zeta Alpha")
    assert n["token_sorted"] == "alpha zeta"


def test_name_repeated_whitespace_collapsed():
    n = normalize_business_name("A    B")
    assert n["exact_normalized"] == "a b"


def test_name_empty_and_nan_safe():
    for val in (None, "", "nan", "NaN"):
        n = normalize_business_name(val)
        assert n["exact_normalized"] == ""
        assert n["tokens"] == []


def test_address_keeps_numbers():
    a = normalize_address("123 Main St")
    assert "123" in a["address_normalized"]
    assert a["house_number"] == "123"


def test_address_street_abbreviation_expansion():
    a = normalize_address("123 Main St")
    assert "street" in a["address_normalized"]


def test_address_unit_extraction():
    a = normalize_address("123 Main St Apt 4B")
    assert a["unit_token"] == "4b"
    assert "4b" not in a["street_tokens"]


def test_address_postal_code_extraction_generic():
    a = normalize_address("123 Main Street 94105")
    assert a["postal_code"] == "94105"


def test_address_does_not_assume_us_format_france():
    a = normalize_address("45 Rue de la Paix 75002")
    assert a["house_number"] == "45"
    assert a["postal_code"] == "75002"
    assert "rue" in a["street_tokens"]


def test_address_alnum_has_no_spaces():
    a = normalize_address("123 Main St, Suite 400")
    assert " " not in a["address_alnum"]


def test_country_alias_collapse():
    c1 = normalize_country("USA")
    c2 = normalize_country("United States")
    assert c1["normalized"] == c2["normalized"] == "united states"


def test_country_not_hardcoded_to_us_india_only():
    c = normalize_country("France")
    assert c["normalized"] == "france"


def test_country_empty_safe():
    c = normalize_country(None)
    assert c["normalized"] == ""


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
