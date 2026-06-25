import os
from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np

from .artifacts import ArtifactSet, find_local_artifacts, load_json_map
from .catalog import prefer_catalog
from .errors import ModelLoadError, ValidationError
from .models import get_model_spec, normalize_catalog
from .preprocess import preprocess_points, validate_request


DEFAULT_BATCH_SIZE = 256


def _softmax_max_prob(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    predicted_ids = np.argmax(probabilities, axis=-1)
    max_probabilities = np.take_along_axis(
        probabilities, predicted_ids[..., np.newaxis], axis=-1
    )[..., 0]
    return predicted_ids, max_probabilities


class StarSolver:
    def __init__(
        self,
        model: str,
        model_dir: str | os.PathLike | None = None,
        artifact_set: ArtifactSet | None = None,
        session=None,
        class_id_to_catalog_id: dict[str, str] | None = None,
        overlap_map: dict[str, str] | None = None,
    ):
        self.model = get_model_spec(model).name
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.artifact_set = artifact_set
        self.load_timings = {}

        needs_artifacts = (
            self.artifact_set is None
            and (session is None or class_id_to_catalog_id is None or overlap_map is None)
        )
        if needs_artifacts:
            artifact_started = time.perf_counter()
            self.artifact_set = find_local_artifacts(self.model, model_dir=self.model_dir)
            self.load_timings["local_artifact_discovery_seconds"] = (
                time.perf_counter() - artifact_started
            )

        session_started = time.perf_counter()
        self.session = session if session is not None else self._create_session()
        self.load_timings["onnx_session_seconds"] = (
            0.0 if session is not None else time.perf_counter() - session_started
        )

        map_started = time.perf_counter()
        self.class_id_to_catalog_id = (
            {str(key): str(value) for key, value in class_id_to_catalog_id.items()}
            if class_id_to_catalog_id is not None
            else load_json_map(self.artifact_set.star_id_map_path)
        )
        self.class_id_to_catalog_id["0"] = ""
        self.load_timings["map_load_seconds"] = time.perf_counter() - map_started

        overlap_started = time.perf_counter()
        self.overlap_map = (
            {str(key): str(value) for key, value in overlap_map.items()}
            if overlap_map is not None
            else load_json_map(self.artifact_set.overlap_path)
        )
        self.load_timings["overlap_load_seconds"] = time.perf_counter() - overlap_started

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _create_session(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelLoadError(
                "onnxruntime is not installed; install it with `pip install zystarsolver` "
                "or `pip install onnxruntime`."
            ) from exc
        try:
            return ort.InferenceSession(
                str(self.artifact_set.onnx_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelLoadError(f"failed to load ONNX model: {exc}") from exc

    def close(self) -> None:
        self.session = None
        self.class_id_to_catalog_id = {}
        self.overlap_map = {}
        self.artifact_set = None

    def _format_results(self, logits: np.ndarray, requested_catalog: str) -> list[dict]:
        predicted_class_ids, probabilities = _softmax_max_prob(logits)
        raw_catalog_ids = [
            str(self.class_id_to_catalog_id.get(str(class_id), ""))
            for class_id in predicted_class_ids.tolist()
        ]
        preferred_ids = prefer_catalog(raw_catalog_ids, requested_catalog, self.overlap_map)
        return [
            {"catalog_id": catalog_id, "probability": float(probability)}
            for catalog_id, probability in zip(preferred_ids, probabilities.tolist())
        ]

    @staticmethod
    def _normalize_points_batch(points_batch) -> list:
        if not isinstance(points_batch, Sequence) or isinstance(points_batch, (str, bytes)):
            raise ValidationError("points_batch must be a non-empty list of point lists")
        normalized = list(points_batch)
        if not normalized:
            raise ValidationError("points_batch must be a non-empty list of point lists")
        return normalized

    @staticmethod
    def _normalize_batch_size(batch_size) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValidationError("batch_size must be a positive integer")
        return batch_size

    def predict(self, width, height, points, catalog: str = "HIP") -> dict[str, list[dict]]:
        if self.session is None:
            raise ModelLoadError("this StarSolver has been closed; create a new StarSolver")

        requested_catalog = normalize_catalog(catalog)
        validate_request(self.model, width, height, points, requested_catalog)
        features, num_points = preprocess_points(self.model, width, height, points)
        input_array = np.asarray([features], dtype=np.float32)
        logits = self.session.run(["logits"], {"features": input_array})[0]
        result_logits = np.asarray(logits)[0, :num_points, :]
        results = self._format_results(result_logits, requested_catalog)
        return {"results": results}

    def predict_batch(
        self,
        width,
        height,
        points_batch,
        catalog: str = "HIP",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, list[list[dict]]]:
        if self.session is None:
            raise ModelLoadError("this StarSolver has been closed; create a new StarSolver")

        requested_catalog = normalize_catalog(catalog)
        normalized_batch = self._normalize_points_batch(points_batch)
        normalized_batch_size = self._normalize_batch_size(batch_size)
        results = []
        for batch_start in range(0, len(normalized_batch), normalized_batch_size):
            batch_points = normalized_batch[batch_start : batch_start + normalized_batch_size]
            features_batch = []
            num_points_batch = []
            for points in batch_points:
                features, num_points = preprocess_points(self.model, width, height, points)
                features_batch.append(features)
                num_points_batch.append(num_points)

            input_array = np.asarray(features_batch, dtype=np.float32)
            logits = np.asarray(self.session.run(["logits"], {"features": input_array})[0])
            results.extend(
                self._format_results(logits[index, :num_points, :], requested_catalog)
                for index, num_points in enumerate(num_points_batch)
            )
        return {"results": results}
