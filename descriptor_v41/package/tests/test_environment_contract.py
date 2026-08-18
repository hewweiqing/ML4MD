from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def locked_versions():
    values = {}
    for line in (ROOT / "DELFBLUE_CU118_CONSTRAINTS.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split("==", 1)
            values[name] = value
    return values


def test_critical_alignn_dependency_set_is_explicitly_locked():
    locked = locked_versions()
    assert locked["matminer"] == "0.9.3"
    assert locked["monty"] == "2025.3.3"
    assert locked["pymatgen"] == "2025.10.7"
    assert locked["pyparsing"] == "2.4.7"
    assert locked["flake8"] == "7.3.0"
    assert locked["pycodestyle"] == "2.14.0"
    assert locked["pydocstyle"] == "6.3.0"
    assert int(locked["pyparsing"].split(".")[0]) < 3


def test_setup_uses_hash_locked_transactions_and_pip_check():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert setup.count("--require-hashes") >= 4
    assert 'PYTHON_DEPENDENCY_LOCK.txt' in setup
    assert 'pip install \'matminer==0.9.3\' \'monty==2025.3.3\'' not in setup
    assert '"$PY" -m pip check' in setup
    assert "partial or uncertified v9 prefix rejected" in setup


def test_environment_verifier_declares_same_critical_versions():
    verifier = (ROOT / "scripts" / "verify_environment.py").read_text(encoding="utf-8")
    for name, value in locked_versions().items():
        if name in {"flake8", "jarvis-tools", "matminer", "monty", "numpy", "pycodestyle", "pydocstyle", "pymatgen", "pyparsing"}:
            assert f'"{name}": "{value}"' in verifier
