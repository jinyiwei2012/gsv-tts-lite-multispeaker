import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lifespan_is_defined_before_fastapi_app_creation():
    for relative_path in (
        "API/fastapi_server_example.py",
        "API/personal_api.py",
    ):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        lifespan_line = next(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
        )
        app_line = next(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "FastAPI"
        )

        assert lifespan_line < app_line, relative_path
