import json
from pathlib import Path

import pytest

from ZYstarsolver.artifacts import find_local_artifacts
from ZYstarsolver.errors import ModelLoadError


def write_model_files(root: Path, model: str = "30-100", *, overlap: bool = True) -> Path:
    model_dir = root / model
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"not a real onnx file")
    (model_dir / "star_id_map.json").write_text(json.dumps({"1": "HIP1"}), encoding="utf-8")
    if overlap:
        (root / "overlap_10.json").write_text(json.dumps({"GAIA1": "HIP1"}), encoding="utf-8")
    return model_dir


def test_find_local_artifacts_uses_explicit_model_dir(tmp_path):
    model_dir = write_model_files(tmp_path)

    artifacts = find_local_artifacts("30-100", model_dir=model_dir)

    assert artifacts.onnx_path == model_dir / "model.onnx"
    assert artifacts.star_id_map_path == model_dir / "star_id_map.json"
    assert artifacts.overlap_path == tmp_path / "overlap_10.json"


def test_find_local_artifacts_uses_environment_model_dir(tmp_path, monkeypatch):
    model_dir = write_model_files(tmp_path, "10-35")
    monkeypatch.setenv("ZYSTARSOLVER_MODEL_DIR", str(tmp_path))

    artifacts = find_local_artifacts("10-35")

    assert artifacts.onnx_path == model_dir / "model.onnx"


@pytest.mark.parametrize(
    ("missing_file", "expected"),
    [
        ("onnx", ".onnx"),
        ("star_id_map", "star_id_map.json"),
        ("overlap", "overlap_10.json"),
    ],
)
def test_find_local_artifacts_reports_missing_files_with_download_command(
    tmp_path, missing_file, expected
):
    model_dir = write_model_files(tmp_path, overlap=missing_file != "overlap")
    if missing_file == "onnx":
        (model_dir / "model.onnx").unlink()
    elif missing_file == "star_id_map":
        (model_dir / "star_id_map.json").unlink()

    with pytest.raises(ModelLoadError) as exc:
        find_local_artifacts("30-100", model_dir=model_dir)

    message = str(exc.value)
    assert "30-100" in message
    assert expected in message
    assert "python scripts/download_models.py --model 30-100" in message
    assert str(model_dir) in message
