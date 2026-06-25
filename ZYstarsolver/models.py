from dataclasses import dataclass

from .errors import ValidationError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    min_points: int
    max_points: int
    max_seq_len: int


MODEL_SPECS = {
    "10-35": ModelSpec(name="10-35", min_points=25, max_points=40, max_seq_len=40),
    "30-100": ModelSpec(name="30-100", min_points=20, max_points=30, max_seq_len=30),
}


def get_model_spec(model: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(model)]
    except KeyError as exc:
        raise ValidationError("model must be '10-35' or '30-100'") from exc


def normalize_catalog(catalog: str) -> str:
    normalized = str(catalog or "HIP").upper()
    if normalized not in {"HIP", "GAIA"}:
        raise ValidationError("catalog must be 'HIP' or 'GAIA'")
    return normalized
