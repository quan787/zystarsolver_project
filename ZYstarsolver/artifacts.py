import json
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelLoadError
from .models import get_model_spec


@dataclass(frozen=True)
class ArtifactSet:
    onnx_path: Path
    star_id_map_path: Path
    overlap_path: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_models_root() -> Path:
    return _project_root() / "models"


def _download_hint(model: str) -> str:
    return f"python scripts/download_models.py --model {model}"


def _format_model_error(model: str, checked_paths: list[Path], missing: list[str]) -> ModelLoadError:
    checked = "\n".join(f"  - {path}" for path in checked_paths)
    missing_text = ", ".join(missing)
    return ModelLoadError(
        f"model artifacts for {model} are incomplete; missing {missing_text}.\n"
        f"Checked paths:\n{checked}\n"
        f"Run `{_download_hint(model)}` from the project root to download the model files."
    )


def _candidate_model_dirs(model: str, model_dir: str | os.PathLike | None = None) -> list[Path]:
    if model_dir is not None:
        explicit = Path(model_dir)
        return [explicit if explicit.name == model else explicit / model]

    candidates = []
    env_root = os.environ.get("ZYSTARSOLVER_MODEL_DIR")
    if env_root:
        candidates.append(Path(env_root) / model)
    candidates.append(default_models_root() / model)
    candidates.append(Path.cwd() / "models" / model)
    return candidates


def _find_overlap_path(model_dir: Path) -> tuple[Path | None, list[Path]]:
    checked = []
    env_path = os.environ.get("ZYSTARSOLVER_OVERLAP_PATH")
    if env_path:
        checked.append(Path(env_path))
    checked.extend(
        [
            model_dir / "overlap_10.json",
            model_dir.parent / "overlap_10.json",
            default_models_root() / "overlap_10.json",
        ]
    )
    for candidate in checked:
        if candidate.exists():
            return candidate, checked
    return None, checked


def find_local_artifacts(model: str, model_dir: str | os.PathLike | None = None) -> ArtifactSet:
    spec = get_model_spec(model)
    checked_model_dirs = _candidate_model_dirs(spec.name, model_dir)
    incomplete_errors: list[tuple[Path, list[str]]] = []

    for candidate in checked_model_dirs:
        missing = []
        onnx_files = sorted(candidate.glob("*.onnx")) if candidate.exists() else []
        if not onnx_files:
            missing.append(".onnx")
        star_id_map = candidate / "star_id_map.json"
        if not star_id_map.exists():
            missing.append("star_id_map.json")
        overlap_path, overlap_checked = _find_overlap_path(candidate)
        if overlap_path is None:
            missing.append("overlap_10.json")
        if not missing:
            return ArtifactSet(
                onnx_path=onnx_files[0],
                star_id_map_path=star_id_map,
                overlap_path=overlap_path,
            )
        incomplete_errors.append((candidate, missing))

    all_checked = []
    all_missing = []
    for candidate, missing in incomplete_errors:
        all_checked.append(candidate)
        all_missing.extend(missing)
    if checked_model_dirs:
        _, overlap_checked = _find_overlap_path(checked_model_dirs[0])
        all_checked.extend(overlap_checked)
    unique_missing = list(dict.fromkeys(all_missing or [".onnx", "star_id_map.json", "overlap_10.json"]))
    raise _format_model_error(spec.name, all_checked, unique_missing)


def load_json_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}
