from pathlib import Path

from alignn_stage2.common import package_manifest, verify_manifest, write_json


def _build_toy_package(root: Path) -> None:
    (root / "alignn_stage2").mkdir(parents=True)
    (root / "alignn_stage2" / "__init__.py").write_text("", encoding="utf-8")
    (root / "alignn_stage2" / "thing.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "thing.cpython-310.pyc").write_bytes(b"\x00\x01")


def test_manifest_round_trip_passes(tmp_path: Path):
    root = tmp_path / "toy_package"
    _build_toy_package(root)
    manifest = package_manifest(root)
    write_json(root / "PACKAGE_MANIFEST.json", manifest)
    report = verify_manifest(root)
    assert report["status"] == "passed"
    assert report["missing_declared_files"] == []
    assert report["unexpected_immutable_files"] == []
    assert not any("pycache" in entry["path"] for entry in manifest["files"])


def test_manifest_detects_tampering(tmp_path: Path):
    root = tmp_path / "toy_package"
    _build_toy_package(root)
    manifest = package_manifest(root)
    write_json(root / "PACKAGE_MANIFEST.json", manifest)
    (root / "README.md").write_text("tampered\n", encoding="utf-8")
    report = verify_manifest(root)
    assert report["status"] == "failed"
    assert any(entry["path"] == "README.md" for entry in report["individual_file_mismatches"])


def test_manifest_detects_missing_file(tmp_path: Path):
    root = tmp_path / "toy_package"
    _build_toy_package(root)
    manifest = package_manifest(root)
    write_json(root / "PACKAGE_MANIFEST.json", manifest)
    (root / "README.md").unlink()
    report = verify_manifest(root)
    assert report["status"] == "failed"
    assert "README.md" in report["missing_declared_files"]
