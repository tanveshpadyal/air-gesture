"""Face + age estimation utilities using OpenCV DNN models."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


PointBox = Tuple[int, int, int, int]
AgePrediction = Tuple[PointBox, str, float]

FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
FACE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
AGE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
    "AgeGender/age_deploy.prototxt"
)
AGE_MODEL_URL = (
    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
    "models/age_net.caffemodel"
)


class AgeDetector:
    """
    Detect faces and estimate age group from webcam frames.
    Models are downloaded automatically to `air_gesture_control/models`.
    """

    AGE_BUCKETS = [
        "(0-2)",
        "(4-6)",
        "(8-12)",
        "(15-20)",
        "(25-32)",
        "(38-43)",
        "(48-53)",
        "(60-100)",
    ]
    AGE_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(
        self,
        enabled: bool = False,
        face_confidence: float = 0.7,
        inference_interval: int = 3,
    ) -> None:
        self.enabled = enabled
        self.face_confidence = max(0.1, min(face_confidence, 0.99))
        self.inference_interval = max(1, inference_interval)

        self._face_net = None
        self._age_net = None
        self._frame_counter = 0
        self._last_predictions: List[AgePrediction] = []
        self.warning = ""

        self._models_dir = Path(__file__).resolve().parent / "models"
        self._face_proto_path = self._models_dir / "face_deploy.prototxt"
        self._face_model_path = self._models_dir / "face_detector.caffemodel"
        self._age_proto_path = self._models_dir / "age_deploy.prototxt"
        self._age_model_path = self._models_dir / "age_net.caffemodel"

        if self.enabled:
            self._ensure_loaded()

    def toggle(self) -> bool:
        """Toggle detector mode. Returns new enabled state."""
        self.enabled = not self.enabled
        if self.enabled:
            self._ensure_loaded()
        return self.enabled

    def _download_if_missing(self, path: Path, url: str) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Failed to download model: {path.name}. Cause: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if self._face_net is not None and self._age_net is not None:
            return
        self.warning = ""
        try:
            self._download_if_missing(self._face_proto_path, FACE_PROTO_URL)
            self._download_if_missing(self._face_model_path, FACE_MODEL_URL)
            self._download_if_missing(self._age_proto_path, AGE_PROTO_URL)
            self._download_if_missing(self._age_model_path, AGE_MODEL_URL)

            self._face_net = cv2.dnn.readNetFromCaffe(
                str(self._face_proto_path),
                str(self._face_model_path),
            )
            self._age_net = cv2.dnn.readNetFromCaffe(
                str(self._age_proto_path),
                str(self._age_model_path),
            )
        except Exception as exc:
            self.enabled = False
            self.warning = str(exc)

    def _predict(self, frame_bgr: np.ndarray) -> List[AgePrediction]:
        if self._face_net is None or self._age_net is None:
            return []

        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        predictions: List[AgePrediction] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.face_confidence:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            pad = 18
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue

            age_blob = cv2.dnn.blobFromImage(
                face,
                scalefactor=1.0,
                size=(227, 227),
                mean=self.AGE_MEAN_VALUES,
                swapRB=False,
                crop=False,
            )
            self._age_net.setInput(age_blob)
            age_preds = self._age_net.forward()[0]
            age_idx = int(np.argmax(age_preds))
            age_label = self.AGE_BUCKETS[age_idx]
            age_conf = float(age_preds[age_idx])

            predictions.append(((x1, y1, x2, y2), age_label, age_conf))
        return predictions

    def process_and_draw(self, frame_bgr: np.ndarray) -> int:
        """Run age estimation and draw face boxes/labels. Returns face count."""
        if not self.enabled:
            return 0

        self._ensure_loaded()
        if not self.enabled:
            return 0

        self._frame_counter += 1
        if self._frame_counter % self.inference_interval == 0 or not self._last_predictions:
            self._last_predictions = self._predict(frame_bgr)

        for (x1, y1, x2, y2), age_label, age_conf in self._last_predictions:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (40, 220, 220), 2)
            text = f"Age {age_label} {age_conf:.2f}"
            cv2.putText(
                frame_bgr,
                text,
                (x1, max(18, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 255, 255),
                2,
            )
        return len(self._last_predictions)
