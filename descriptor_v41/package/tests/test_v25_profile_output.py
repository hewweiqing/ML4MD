import ast
from pathlib import Path


def test_profile_report_path_and_model_tensor_have_distinct_bindings():
    """Regression for the post-batch failure in DelftBlue job 10633397."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/profile_100_batches.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    stored_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                stored_names.extend(
                    child.id for child in ast.walk(target)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                )
    assert "output_path" in stored_names
    assert "model_output" in stored_names
    assert "output" not in stored_names
    assert "output_path.parent.mkdir(parents=True, exist_ok=True)" in source
    assert "os.replace(temporary, output_path)" in source
