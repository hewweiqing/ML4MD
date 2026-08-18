from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verify_environment_is_self_bootstrapping_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_environment.py"), "--help"],
        cwd=ROOT.parent, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--expected-prefix" in result.stdout


def test_every_setup_environment_verifier_call_has_explicit_package_root():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    calls = setup.count('"$PACKAGE_DIR/scripts/verify_environment.py"')
    prefixes = setup.count('PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"')
    assert calls == 3
    assert prefixes >= calls + 2


def test_slurm_entrypoints_export_submit_directory_pythonpath():
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        source = path.read_text(encoding="utf-8")
        assert 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"' in source
