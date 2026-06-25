from pathlib import Path

import ZYstarsolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_package_exports_only_star_solver():
    removed_predict_function = "predict" + "_stars"
    removed_cache_function = "clear" + "_solver_cache"

    assert ZYstarsolver.__all__ == ["StarSolver"]
    assert hasattr(ZYstarsolver, "StarSolver")
    assert not hasattr(ZYstarsolver, removed_predict_function)
    assert not hasattr(ZYstarsolver, removed_cache_function)


def test_readme_is_english_first_then_chinese_without_internal_notes():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    removed_predict_function = "predict" + "_stars"
    removed_cache_function = "clear" + "_solver_cache"
    removed_api_sample_line = "Run the included " + "API samples"
    removed_model_section = "Model" + " Files"
    removed_training_sidecar = "star" + "_cartesian"

    assert readme.startswith("# ZYstarsolver\n\n[中文](#中文)")
    assert "\n## 中文\n" in readme
    assert "Quick Start: Hosted API" in readme.split("\n## 中文\n", 1)[0]
    assert "快速上手" in readme.split("\n## 中文\n", 1)[1]
    assert removed_api_sample_line not in readme
    assert removed_model_section not in readme
    assert removed_training_sidecar not in readme
    assert removed_predict_function not in readme
    assert removed_cache_function not in readme
    assert "git+ssh://git@github.com/quan787/zystarsolver_project.git" in readme
