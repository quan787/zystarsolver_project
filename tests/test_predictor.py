import numpy as np
import pytest

from ZYstarsolver import StarSolver
from ZYstarsolver.errors import ModelLoadError
from ZYstarsolver.predictor import _softmax_max_prob


class FakeSession:
    def __init__(self, class_ids):
        self.class_ids = class_ids
        self.run_count = 0

    def run(self, output_names, feed):
        self.run_count += 1
        features = feed["features"]
        batch, total_features = features.shape
        seq_len = total_features // 4
        vocab_size = max(self.class_ids) + 1
        logits = np.full((batch, seq_len, vocab_size), -20.0, dtype=np.float32)
        for index, class_id in enumerate(self.class_ids):
            logits[0, index, class_id] = 20.0
        return [logits]


def test_softmax_returns_predicted_id_and_probability():
    logits = np.asarray([[[0.0, 3.0], [4.0, 0.0]]], dtype=np.float32)

    class_ids, probabilities = _softmax_max_prob(logits)

    assert class_ids.tolist() == [[1, 0]]
    assert probabilities[0, 0] > 0.95
    assert probabilities[0, 1] > 0.98


def test_star_solver_predict_returns_catalog_id_and_probability():
    solver = StarSolver(
        model="30-100",
        session=FakeSession([1, 2] + [0] * 28),
        class_id_to_catalog_id={"0": "", "1": "GAIA123", "2": "HIP42"},
        overlap_map={"GAIA123": "HIP1"},
    )
    points = [(100.0, 100.0)] * 20

    response = solver.predict(width=1920, height=1200, points=points, catalog="HIP")

    assert set(response) == {"results"}
    assert response["results"][:2] == [
        {"catalog_id": "HIP1", "probability": 1.0},
        {"catalog_id": "HIP42", "probability": 1.0},
    ]
    assert set(response["results"][0]) == {"catalog_id", "probability"}
    assert len(response["results"]) == 20
    assert solver.session.run_count == 1


def test_star_solver_close_releases_session_and_rejects_further_prediction():
    solver = StarSolver(
        model="30-100",
        session=FakeSession([0] * 30),
        class_id_to_catalog_id={"0": ""},
        overlap_map={},
    )

    solver.close()

    assert solver.session is None
    with pytest.raises(ModelLoadError, match="closed"):
        solver.predict(width=1920, height=1200, points=[(10.0, 10.0)] * 20)


def test_star_solver_context_manager_closes_session():
    with StarSolver(
        model="30-100",
        session=FakeSession([0] * 30),
        class_id_to_catalog_id={"0": ""},
        overlap_map={},
    ) as solver:
        assert solver.session is not None

    assert solver.session is None
