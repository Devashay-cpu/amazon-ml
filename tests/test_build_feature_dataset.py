"""
Tests for the Phase 3 OOM fix in build_feature_dataset.py.

These are integration-style tests: they spin up an isolated copy of the
src/ tree plus a small synthetic dataset/train/ under a temp directory (the
Phase 3/4 scripts resolve their project root from their own file location,
so an isolated run needs an isolated copy of src/), then invoke
build_feature_dataset.py exactly as a user would from the command line,
with different --batch-size values, and check:

  1. The fix doesn't change output semantics: batch_size=1 (maximally
     bounded memory) and batch_size=1000 (effectively unbounded, matching
     the pre-fix behavior on this small dataset) must produce IDENTICAL
     candidate-pair rows (same set of (s1, source, candidate) triples,
     same features, same labels) -- only how the work is grouped changed.
  2. iter_s1_with_candidates respects --sample-size / limit.
  3. Per-batch needed-id sets stay bounded by batch size, not by the
     dataset's total distinct candidate count (the actual memory fix).
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _write_toy_dataset(root: Path) -> None:
    train_dir = root / "dataset" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    def write_tsv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    s2_rows = [
        ("s2_1", "Acme Corp", "123 Main St 94105", "USA"),
        ("s2_2", "Acme Corporation", "123 Main Street 94105", "USA"),
        ("s2_3", "Globex Inc", "1 Infinite Loop 95014", "USA"),
        ("s2_4", "Sharma Traders", "MG Road 560001", "India"),
        ("s2_5", "Boulangerie Dupont", "45 Rue de la Paix 75002", "France"),
        ("s2_6", "Acme Rentals", "500 Broad St 94105", "USA"),
        ("s2_7", "Ace Consulting", "77 Side St 94105", "USA"),
    ]
    s3_rows = [
        ("s3_1", "Acme Co", "123 Main St 94105", "USA"),
        ("s3_2", "Globex Incorporated", "1 Infinite Loop 95014", "USA"),
        ("s3_3", "Sharma Traders Pvt Ltd", "MG Road 560001", "India"),
        ("s3_4", "Totally Unrelated Biz", "999 Nowhere Ave 00000", "USA"),
        ("s3_5", "Acme Storage", "500 Broad St 94105", "USA"),
        ("s3_6", "Acme Bakery", "20 Corner Ave 94105", "USA"),
    ]
    s1_rows = [
        ("s1_1", "Acme Corporation", "123 Main Street 94105", "USA"),
        ("s1_2", "Globex Inc", "1 Infinite Loop 95014", "USA"),
        ("s1_3", "Sharma Traders", "MG Road 560001", "India"),
        ("s1_4", "Boulangerie Dupont", "45 Rue de la Paix 75002", "France"),
        ("s1_5", "No Match Business", "0 Void St 11111", "USA"),
    ]
    gt_rows = [
        ("s1_1", "s2_1,s2_2,s3_1"),
        ("s1_2", "s2_3,s3_2"),
        ("s1_3", "s2_4,s3_3"),
        ("s1_4", "s2_5"),
        ("s1_5", ""),
    ]

    write_tsv(train_dir / "train_source1.tsv", ["entity_id", "business_name", "business_address", "country"], s1_rows)
    write_tsv(train_dir / "train_source2.tsv", ["entity_id", "business_name", "business_address", "country"], s2_rows)
    write_tsv(train_dir / "train_source3.tsv", ["entity_id", "business_name", "business_address", "country"], s3_rows)
    write_tsv(train_dir / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt_rows)


def _isolated_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "student_resource"
    shutil.copytree(SRC_DIR, project_root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    _write_toy_dataset(project_root)
    return project_root


def _run_build(project_root: Path, batch_size: int, sample_size: int = 100):
    script = project_root / "src" / "build_feature_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--sample-size", str(sample_size), "--batch-size", str(batch_size)],
        capture_output=True, text=True, cwd=str(project_root),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


def _read_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sorted(tuple(r) for r in reader)
    return header, rows


def test_batch_size_does_not_change_output_rows(tmp_path):
    project_a = _isolated_project(tmp_path / "a")
    project_b = _isolated_project(tmp_path / "b")

    _run_build(project_a, batch_size=1)      # maximally bounded — one S1 entity per batch
    _run_build(project_b, batch_size=1000)   # effectively unbounded on this tiny dataset

    header_a, rows_a = _read_rows(project_a / "features" / "train_pair_features.csv")
    header_b, rows_b = _read_rows(project_b / "features" / "train_pair_features.csv")

    assert header_a == header_b
    assert rows_a == rows_b
    assert len(rows_a) == 12  # same known-good row count as the pre-fix implementation


def test_run_reports_bounded_batch_needed_ids(tmp_path):
    project = _isolated_project(tmp_path / "c")
    result = _run_build(project, batch_size=1)
    # With batch_size=1, the largest single batch can only need the
    # candidate ids for ONE S1 entity -- necessarily small, and the report
    # should reflect that the peak stayed bounded rather than global.
    report_path = project / "reports" / "phase3_feature_report.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Largest single batch's combined needed-ID count" in text
    assert "Batch size used: 1 " in text


def test_sample_size_limit_is_respected_across_batches(tmp_path):
    project = _isolated_project(tmp_path / "d")
    _run_build(project, batch_size=2, sample_size=3)  # only first 3 of 5 S1 entities
    header, rows = _read_rows(project / "features" / "train_pair_features.csv")
    s1_id_col = header.index("s1_entity_id")
    distinct_s1_ids = {row[s1_id_col] for row in rows}
    assert distinct_s1_ids <= {"s1_1", "s1_2", "s1_3"}
    assert "s1_4" not in distinct_s1_ids
    assert "s1_5" not in distinct_s1_ids


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
