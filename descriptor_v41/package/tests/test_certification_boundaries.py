from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.require_certification import main as require_main


ROOT = Path(__file__).resolve().parents[1]


def test_login_verifier_imports_dgl_and_alignn_only_after_runtime_inspection():
    source = (ROOT / "scripts" / "verify_environment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "dgl" not in imported
    assert "alignn" not in imported
    assert 'runtime = inspect_runtime(Path(expected_prefix))' in source
    assert source.index('runtime = inspect_runtime(Path(expected_prefix))') < source.index('for module_name in ("torch", "dgl", "alignn")')
    assert '"gpu_required_for_this_check": False' in source


def test_a100_preflight_records_all_required_runtime_evidence():
    source = (ROOT / "scripts" / "a100_preflight.py").read_text(encoding="utf-8")
    for token in ("hostname", "gpu_model", "nvidia_smi", "nvidia_driver_version",
            "torch_cuda_runtime", "torch_cuda_is_available", "libcusparse_path", "dgl_version",
            "alignn_model_constructed_on_cuda", "gpu_tensor_operation", "a100_runtime_certified"):
        assert token in source


def test_scientific_slurm_scripts_require_a100_certification():
    for name in ("10_train_fold_seed.sbatch", "20_calibrate_export.sbatch"):
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "require_v37_external_gates.sh" in source
    helper = (ROOT / "scripts/require_v37_external_gates.sh").read_text(encoding="utf-8")
    assert "require_certification.py" in helper and "--kind a100" in helper
    assert "$V26_ROOT/preflight/A100_RUNTIME_CERTIFICATION.json" in helper
    source = (ROOT / "slurm/30_consolidate_oof.sbatch").read_text(encoding="utf-8")
    assert "require_certification.py --kind a100" in source
    assert "A100_RUNTIME_CERTIFICATION.json" in source
