import json
import zipfile
from pathlib import Path

import pytest

from ZYstarsolver.download_models import main


def write_archive(path: Path, model: str = "30-100", *, include_overlap: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{model}/model.onnx", b"not a real onnx file")
        archive.writestr(f"{model}/star_id_map.json", json.dumps({"1": "HIP1"}))
        if include_overlap:
            archive.writestr("overlap_10.json", json.dumps({"GAIA1": "HIP1"}))


def test_url_requires_model():
    with pytest.raises(SystemExit, match="--url requires --model"):
        main(["--url", "https://example.com/model.zip"])


def test_archive_requires_model(tmp_path):
    archive_path = tmp_path / "model.zip"
    write_archive(archive_path)

    with pytest.raises(SystemExit, match="--archive requires --model"):
        main(["--archive", str(archive_path)])


def test_url_and_archive_cannot_be_used_together(tmp_path):
    archive_path = tmp_path / "model.zip"
    write_archive(archive_path)

    with pytest.raises(SystemExit, match="--url and --archive cannot be used together"):
        main(
            [
                "--model",
                "30-100",
                "--url",
                "https://example.com/model.zip",
                "--archive",
                str(archive_path),
            ]
        )


def test_local_archive_extracts_requested_model(tmp_path):
    archive_path = tmp_path / "model.zip"
    model_root = tmp_path / "models"
    write_archive(archive_path)

    assert main(["--model", "30-100", "--archive", str(archive_path), "--model-dir", str(model_root)]) == 0

    assert (model_root / "30-100" / "model.onnx").exists()
    assert (model_root / "30-100" / "star_id_map.json").exists()
    assert (model_root / "overlap_10.json").exists()


def test_local_archive_reports_missing_overlap(tmp_path):
    archive_path = tmp_path / "model.zip"
    model_root = tmp_path / "models"
    write_archive(archive_path, include_overlap=False)

    with pytest.raises(SystemExit, match="overlap_10.json"):
        main(["--model", "30-100", "--archive", str(archive_path), "--model-dir", str(model_root)])
