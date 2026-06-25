import math
from collections.abc import Sequence

from .errors import ValidationError
from .models import ModelSpec, get_model_spec, normalize_catalog


def _as_positive_float(value, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive number") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValidationError(f"{name} must be a positive number")
    return converted


def _normalize_points(points) -> list[tuple[float, float]]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValidationError("points must be a list of [x, y] pairs")

    normalized = []
    for index, point in enumerate(points):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValidationError(f"point {index} must be an [x, y] pair")
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"point {index} coordinates must be numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValidationError(f"point {index} coordinates must be finite numbers")
        normalized.append((x, y))
    return normalized


def validate_request(
    model: str,
    width,
    height,
    points,
    catalog: str = "HIP",
) -> ModelSpec:
    spec = get_model_spec(model)
    image_width = _as_positive_float(width, "width")
    image_height = _as_positive_float(height, "height")
    normalize_catalog(catalog)
    normalized_points = _normalize_points(points)

    if not (spec.min_points <= len(normalized_points) <= spec.max_points):
        raise ValidationError(
            f"model {spec.name} requires {spec.min_points}-{spec.max_points} input stars"
        )

    for index, (x, y) in enumerate(normalized_points):
        if x < 0 or x > image_width or y < 0 or y > image_height:
            raise ValidationError(f"point {index} is outside image bounds")

    return spec


def preprocess_points(model: str, width, height, points) -> tuple[list[float], int]:
    spec = validate_request(model, width, height, points, "HIP")
    image_width = float(width)
    image_height = float(height)
    normalized_points = _normalize_points(points)

    features: list[float] = []
    for x_pixel, y_pixel in normalized_points[: spec.max_seq_len]:
        norm_x = (x_pixel - image_width / 2.0) / image_width
        norm_y = (y_pixel - image_height / 2.0) / image_width
        radius = math.sqrt(norm_x**2 + norm_y**2)
        theta = (math.atan2(norm_y, norm_x) + math.pi) / (2.0 * math.pi)
        features.extend([norm_x, norm_y, radius, theta])

    pad_len = spec.max_seq_len * 4 - len(features)
    if pad_len > 0:
        features.extend([-1.0] * pad_len)
    return features, len(normalized_points)
