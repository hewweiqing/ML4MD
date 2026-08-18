from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alignn_stage2.common import aggregate_from_lines, package_manifest, verify_manifest, write_json


def test_manifest_uses_case_sensitive_posix_lexical_order(tmp_path):
    (tmp_path / "z.py").write_text("z\n", encoding="utf-8")
    (tmp_path / "A.json").write_text("A\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    manifest = package_manifest(tmp_path)
    assert [item["path"] for item in manifest["files"]] == ["A.json", "a.py", "z.py"]
    expected_bytes = "".join(f"{item['sha256']}  {item['path']}\n" for item in manifest["files"]).encode("utf-8")
    assert manifest["aggregate_sha256"] == hashlib.sha256(expected_bytes).hexdigest()


def test_manifest_is_not_self_referential_and_generated_material_is_excluded(tmp_path):
    (tmp_path / "source.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "PACKAGE_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "source.cpython-310.pyc").write_bytes(b"generated")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "setup.log").write_text("runtime\n", encoding="utf-8")
    manifest = package_manifest(tmp_path)
    assert [item["path"] for item in manifest["files"]] == ["source.py"]


def test_verifier_reports_missing_unexpected_and_hash_mismatch(tmp_path):
    for name, value in (("a.txt", "a\n"), ("b.txt", "b\n")):
        (tmp_path / name).write_text(value, encoding="utf-8")
    manifest = package_manifest(tmp_path)
    write_json(tmp_path / "PACKAGE_MANIFEST.json", manifest)
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("unexpected\n", encoding="utf-8")
    report = verify_manifest(tmp_path)
    assert report["status"] == "failed"
    assert report["missing_declared_files"] == ["b.txt"]
    assert report["unexpected_immutable_files"] == ["c.txt"]
    assert report["individual_file_mismatches"][0]["path"] == "a.txt"
    assert report["expected_aggregate"] != report["actual_aggregate"]
    assert report["ordered_expected_aggregate_inputs"]
    assert report["ordered_actual_aggregate_inputs"]
    assert report["exclusion_rules"]

