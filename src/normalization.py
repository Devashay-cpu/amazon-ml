"""
Phase 2 — Normalization

Country-agnostic normalization for business_name, business_address, and
country fields. Produces MULTIPLE representations per field (never just one
normalized string), and always keeps the raw value alongside normalized
forms so no information is destroyed.

Design constraints (per spec):
  - Unicode-normalize, lowercase, punctuation/whitespace cleanup.
  - Do NOT aggressively strip numbers (addresses need house numbers,
    postal codes, unit numbers).
  - Do NOT hard-code logic to only US/India-style addresses — France (and
    anything else) must work through the same generic rules.
  - No external geographic data / APIs are used anywhere in this module.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Reference token sets (small, generic, NOT a geographic database — just
# common multinational legal-entity suffixes and English street/unit
# abbreviations that appear across many countries' address formats).
# --------------------------------------------------------------------------

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "gmbh", "srl", "sarl", "sa", "sas", "plc", "llp",
    "pvt", "private", "pte", "kk", "ag", "bv", "nv", "oy", "ab", "spa",
    "group", "holdings", "enterprises", "intl", "international",
}

STREET_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "ter": "terrace", "cir": "circle", "hts": "heights",
    "xing": "crossing", "expy": "expressway", "fwy": "freeway",
}

UNIT_INDICATORS = {
    "apt", "apartment", "suite", "ste", "unit", "fl", "floor", "bldg",
    "building", "rm", "room", "no", "#",
}

_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_WS_RE = re.compile(r"\s+")
_HOUSE_NUM_RE = re.compile(r"^\d+[a-z]?$")
_POSTAL_CANDIDATE_RE = re.compile(r"^[a-z0-9]{3,10}$")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _safe(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    text = str(raw)
    if text.strip().lower() in ("nan", "none", "null"):
        return ""
    return text


# --------------------------------------------------------------------------
# BUSINESS NAME
# --------------------------------------------------------------------------

def normalize_business_name(raw: Optional[str]) -> Dict[str, Any]:
    raw_str = _safe(raw)
    unicode_norm = normalize_unicode(raw_str).strip()
    lower = unicode_norm.lower()
    amp_expanded = lower.replace("&", " and ")

    exact_normalized = clean_whitespace(_NON_ALNUM_SPACE_RE.sub(" ", amp_expanded))
    tokens = exact_normalized.split() if exact_normalized else []

    # Strip trailing legal-suffix tokens (repeatedly, e.g. "x corp ltd" -> "x")
    suffix_tokens = list(tokens)
    while suffix_tokens and suffix_tokens[-1] in LEGAL_SUFFIXES:
        suffix_tokens.pop()
    suffix_stripped = " ".join(suffix_tokens)

    alnum_normalized = _NON_ALNUM_RE.sub("", exact_normalized)
    token_sorted = " ".join(sorted(tokens))
    prefix4 = alnum_normalized[:4]
    suffix4 = alnum_normalized[-4:] if len(alnum_normalized) >= 4 else alnum_normalized

    return {
        "raw": raw_str,
        "unicode_normalized": unicode_norm,
        "exact_normalized": exact_normalized,
        "suffix_stripped": suffix_stripped,
        "alnum_normalized": alnum_normalized,
        "token_sorted": token_sorted,
        "tokens": tokens,
        "prefix4": prefix4,
        "suffix4": suffix4,
    }


# --------------------------------------------------------------------------
# BUSINESS ADDRESS
# --------------------------------------------------------------------------

def normalize_address(raw: Optional[str]) -> Dict[str, Any]:
    raw_str = _safe(raw)
    unicode_norm = normalize_unicode(raw_str).strip()
    lower = unicode_norm.lower()

    # Keep digits — only strip punctuation, do not strip numeric content.
    cleaned = clean_whitespace(_NON_ALNUM_SPACE_RE.sub(" ", lower))
    raw_tokens = cleaned.split() if cleaned else []

    # Expand common street abbreviations token-for-token (never substring).
    expanded_tokens = [STREET_ABBREV.get(t, t) for t in raw_tokens]

    # Extract a unit/suite/apartment token (indicator + following value),
    # removing both from the working token list.
    unit_token = None
    working = []
    skip_next = False
    for i, tok in enumerate(expanded_tokens):
        if skip_next:
            skip_next = False
            continue
        if tok in UNIT_INDICATORS and i + 1 < len(expanded_tokens):
            unit_token = expanded_tokens[i + 1]
            skip_next = True
            continue
        working.append(tok)

    # House number: first token if it looks numeric (digits, optional
    # trailing letter, e.g. "12", "12a") — common across many countries'
    # address conventions, without assuming a single national format.
    house_number = None
    if working and _HOUSE_NUM_RE.match(working[0]):
        house_number = working[0]
        street_candidate_tokens = working[1:]
    else:
        street_candidate_tokens = list(working)

    # Postal code: prefer the LAST token that looks like a postal-code
    # shape (alnum, 3-10 chars, contains a digit) and is not the house
    # number itself. This is a generic heuristic, not a country-specific
    # postal format parser.
    postal_code = None
    remaining_tokens = list(street_candidate_tokens)
    for idx in range(len(remaining_tokens) - 1, -1, -1):
        tok = remaining_tokens[idx]
        if _POSTAL_CANDIDATE_RE.match(tok) and any(ch.isdigit() for ch in tok):
            postal_code = tok
            del remaining_tokens[idx]
            break

    street_tokens = remaining_tokens
    address_normalized = clean_whitespace(" ".join(expanded_tokens))
    address_alnum = _NON_ALNUM_RE.sub("", address_normalized)

    return {
        "raw": raw_str,
        "unicode_normalized": unicode_norm,
        "address_normalized": address_normalized,
        "address_alnum": address_alnum,
        "tokens": expanded_tokens,
        "house_number": house_number,
        "postal_code": postal_code,
        "unit_token": unit_token,
        "street_tokens": street_tokens,
    }


# --------------------------------------------------------------------------
# COUNTRY
# --------------------------------------------------------------------------

# Deliberately tiny and generic — NOT a geographic database. Only collapses
# a handful of extremely common abbreviation/full-name variants so trivial
# formatting differences (periods, "USA" vs "United States") don't split
# otherwise-identical countries into different blocks. Never used to filter,
# restrict, or hard-code which countries are supported.
COUNTRY_ALIASES = {
    "usa": "united states", "u.s.a": "united states", "us": "united states",
    "u.s.": "united states", "united states of america": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom",
    "uae": "united arab emirates",
}


def normalize_country(raw: Optional[str]) -> Dict[str, Any]:
    raw_str = _safe(raw)
    unicode_norm = normalize_unicode(raw_str).strip()
    lower = unicode_norm.lower().replace(".", "")
    cleaned = clean_whitespace(lower)
    normalized = COUNTRY_ALIASES.get(cleaned, cleaned)
    return {
        "raw": raw_str,
        "unicode_normalized": unicode_norm,
        "normalized": normalized,
    }


# --------------------------------------------------------------------------
# COMBINED RECORD NORMALIZATION
# --------------------------------------------------------------------------

def normalize_record(entity_id: str, business_name: Optional[str],
                      business_address: Optional[str], country: Optional[str]) -> Dict[str, Any]:
    """Normalize one source row into the full multi-representation bundle."""
    return {
        "entity_id": entity_id,
        "name": normalize_business_name(business_name),
        "address": normalize_address(business_address),
        "country": normalize_country(country),
    }


if __name__ == "__main__":
    # Quick manual smoke test / demonstration, and generates
    # reports/normalization_report.md from a small built-in example set
    # plus (if available) a live sample from train_source1.tsv.
    import json
    import sys
    import time
    from pathlib import Path

    import pandas as pd

    def find_project_root() -> Path:
        script_path = Path(__file__).resolve()
        candidate = script_path.parent.parent
        if (candidate / "dataset").exists():
            return candidate
        for parent in script_path.parents:
            if (parent / "dataset").exists():
                return parent
        return Path.cwd()

    project_root = find_project_root()
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    example_inputs = [
        ("Acme Corp.", "123 Main St, Suite 400, 94105", "USA"),
        ("ACME  CORPORATION", "123 Main Street Ste 400 94105", "U.S.A"),
        ("Sharma & Sons Pvt. Ltd.", "Flat 12B, MG Road, 560001", "India"),
        ("Boulangerie Dupont", "45 Rue de la Paix, 75002", "France"),
        ("O'Brien's Pub", "10 O'Connell St., Dublin 1", "Ireland"),
    ]

    lines = ["# Normalization Report (Phase 2)\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "## Built-in example transformations\n"]

    for name, addr, country in example_inputs:
        n = normalize_business_name(name)
        a = normalize_address(addr)
        c = normalize_country(country)
        lines.append(f"### Input: name={name!r}, address={addr!r}, country={country!r}\n")
        lines.append(f"- name.exact_normalized = `{n['exact_normalized']}`")
        lines.append(f"- name.suffix_stripped = `{n['suffix_stripped']}`")
        lines.append(f"- name.alnum_normalized = `{n['alnum_normalized']}`")
        lines.append(f"- name.token_sorted = `{n['token_sorted']}`")
        lines.append(f"- address.address_normalized = `{a['address_normalized']}`")
        lines.append(f"- address.house_number = `{a['house_number']}`")
        lines.append(f"- address.postal_code = `{a['postal_code']}`")
        lines.append(f"- address.unit_token = `{a['unit_token']}`")
        lines.append(f"- address.street_tokens = `{a['street_tokens']}`")
        lines.append(f"- country.normalized = `{c['normalized']}`\n")

    # Optional: also sample a handful of real rows if train_source1.tsv exists.
    src1 = project_root / "dataset" / "train" / "train_source1.tsv"
    if src1.exists():
        lines.append("## Sample from train_source1.tsv (first 10 rows)\n")
        df = pd.read_csv(src1, sep="\t", dtype=str, nrows=10)
        for _, row in df.iterrows():
            n = normalize_business_name(row.get("business_name"))
            a = normalize_address(row.get("business_address"))
            c = normalize_country(row.get("country"))
            lines.append(f"- raw_name={row.get('business_name')!r} -> "
                          f"exact=`{n['exact_normalized']}` suffix_stripped=`{n['suffix_stripped']}`")
            lines.append(f"  raw_address={row.get('business_address')!r} -> "
                          f"normalized=`{a['address_normalized']}` house_number=`{a['house_number']}` "
                          f"postal_code=`{a['postal_code']}`")
            lines.append(f"  raw_country={row.get('country')!r} -> normalized=`{c['normalized']}`\n")
    else:
        lines.append(f"\n_(train_source1.tsv not found at {src1}; showing built-in examples only.)_\n")

    out_path = reports_dir / "normalization_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[normalization] Wrote {out_path}")
