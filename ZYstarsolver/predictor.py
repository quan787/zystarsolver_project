import os
from pathlib import Path
import time

import numpy as np

from .artifacts import ArtifactSet, find_local_artifacts, load_json_map
from .catalog import prefer_catalog
from .errors import ModelLoadError
from .models import get_model_spec, normalize_catalog
from .preprocess import preprocess_points, validate_request


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

    def predict(self, width, height, points, catalog: str = "HIP") -> dict[str, list[dict]]:
        if self.session is None:
            raise ModelLoadError("this StarSolver has been closed; create a new StarSolver")

        requested_catalog = normalize_catalog(catalog)
        validate_request(self.model, width, height, points, requested_catalog)
        features, num_points = preprocess_points(self.model, width, height, points)
        input_array = np.asarray([features], dtype=np.float32)
        logits = self.session.run(["logits"], {"features": input_array})[0]
        result_logits = np.asarray(logits)[0, :num_points, :]
        predicted_class_ids, probabilities = _softmax_max_prob(result_logits)
        raw_catalog_ids = [
            str(self.class_id_to_catalog_id.get(str(class_id), ""))
            for class_id in predicted_class_ids.tolist()
        ]
        preferred_ids = prefer_catalog(raw_catalog_ids, requested_catalog, self.overlap_map)
        results = [
            {"catalog_id": catalog_id, "probability": float(probability)}
            for catalog_id, probability in zip(preferred_ids, probabilities.tolist())
        ]
        return {"results": results}
