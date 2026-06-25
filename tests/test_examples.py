import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_infer_help_uses_project_package():
    completed = subprocess.run(
        [sys.executable, "examples/local_infer.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "Run one local ZYstarsolver sample" in completed.stdout
