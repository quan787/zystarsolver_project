from pathlib import Path

import ZYstarsolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_package_exports_only_star_solver():
    assert ZYstarsolver.__all__ == ["StarSolver"]
    assert hasattr(ZYstarsolver, "StarSolver")
    assert not hasattr(ZYstarsolver, "predict_stars")
    assert not hasattr(ZYstarsolver, "clear_solver_cache")


def test_readme_is_english_first_then_chinese_without_internal_notes():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("# ZYstarsolver\n\n[中文](#中文)")
    assert "\n## 中文\n" in readme
    assert "Quick Start: Hosted API" in readme.split("\n## 中文\n", 1)[0]
    assert "快速上手" in readme.split("\n## 中文\n", 1)[1]
    assert "Run the included API samples" not in readme
    assert "Model Files" not in readme
    assert "star_cartesian" not in readme
    assert "predict_stars" not in readme
    assert "clear_solver_cache" not in readme
    assert "git+ssh://git@github.com/quan787/zystarsolver_project.git" in readme
