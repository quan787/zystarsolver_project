import math

import pytest

from ZYstarsolver.errors import ValidationError
from ZYstarsolver.preprocess import preprocess_points, validate_request


def test_preprocess_uses_image_width_for_y_normalization():
    points = [(1024.0, 256.0)] * 25

    features, num_points = preprocess_points("10-35", 2048.0, 1024.0, points)

    assert num_points == 25
    assert features[:4] == pytest.approx([0.0, -0.125, 0.125, 0.25])
    assert len(features) == 40 * 4
    assert features[25 * 4 :] == [-1.0] * ((40 - 25) * 4)


@pytest.mark.parametrize(
    ("model", "count"),
    [
        ("30-100", 19),
        ("30-100", 31),
        ("10-35", 24),
        ("10-35", 41),
    ],
)
def test_validate_request_rejects_wrong_star_counts(model, count):
    points = [(10.0, 10.0)] * count

    with pytest.raises(ValidationError) as exc:
        validate_request(model, 1920, 1200, points, "HIP")

    assert "requires" in str(exc.value)


def test_validate_request_rejects_out_of_frame_coordinates():
    points = [(10.0, 10.0)] * 29 + [(1921.0, 10.0)]

    with pytest.raises(ValidationError) as exc:
        validate_request("30-100", 1920, 1200, points, "HIP")

    assert "outside image bounds" in str(exc.value)


def test_validate_request_rejects_invalid_catalog_and_model():
    points = [(10.0, 10.0)] * 20

    with pytest.raises(ValidationError, match="catalog"):
        validate_request("30-100", 1920, 1200, points, "TYCHO")

    with pytest.raises(ValidationError, match="model"):
        validate_request("20-50", 1920, 1200, points, "HIP")


def test_preprocess_theta_matches_existing_convention():
    points = [(2048.0, 1024.0)] * 25

    features, _ = preprocess_points("10-35", 2048.0, 2048.0, points)

    x, y, radius, theta = features[:4]
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.0)
    assert radius == pytest.approx(0.5)
    assert theta == pytest.approx((math.atan2(0.0, 0.5) + math.pi) / (2 * math.pi))
