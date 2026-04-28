"""MediaPipe hand tracking wrapper with legacy and Tasks API support."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


Point = Tuple[int, int]
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandTracker:
    """Tracks a single hand and returns pixel landmarks."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,
        model_asset_path: Optional[str] = None,
        model_url: str = DEFAULT_MODEL_URL,
    ) -> None:
        self._backend = "legacy" if hasattr(mp, "solutions") else "tasks"
        self._hands = None
        self._last_timestamp_ms = 0
        self._task_connections = []

        if self._backend == "legacy":
            self._setup_legacy(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
            )
        else:
            self._setup_tasks(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_asset_path=model_asset_path,
                model_url=model_url,
            )

    def _setup_legacy(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_complexity: int,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _ensure_task_model(model_asset_path: Optional[str], model_url: str) -> str:
        if model_asset_path:
            path = Path(model_asset_path)
        else:
            path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

        if path.exists():
            return str(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(model_url, timeout=20) as response:
                data = response.read()
            path.write_bytes(data)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download HandLandmarker model file. "
                f"Set --model-asset-path manually or check internet access. Cause: {exc}"
            ) from exc

        return str(path)

    def _setup_tasks(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_asset_path: Optional[str],
        model_url: str,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        resolved_model_path = self._ensure_task_model(model_asset_path=model_asset_path, model_url=model_url)
        base_options = mp_python.BaseOptions(model_asset_path=resolved_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._task_connections = list(vision.HandLandmarksConnections.HAND_CONNECTIONS)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def _draw_task_landmarks(self, frame_bgr, landmarks: List[Point]) -> None:
        for conn in self._task_connections:
            start = landmarks[conn.start]
            end = landmarks[conn.end]
            cv2.line(frame_bgr, start, end, (80, 220, 80), 2)

        for pt in landmarks:
            cv2.circle(frame_bgr, pt, 3, (30, 180, 255), -1)

    def _process_legacy(self, frame_bgr, draw: bool = True) -> Optional[Dict[str, object]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame_bgr.shape

        pixel_landmarks: List[Point] = []
        for lm in hand_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_landmarks.append((px, py))

        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        if draw:
            self._mp_draw.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        return {"landmarks": pixel_landmarks, "handedness": handedness}

    def _process_tasks(self, frame_bgr, draw: bool = True) -> Optional[Dict[str, object]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._hands.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        h, w, _ = frame_bgr.shape
        landmark_list = result.hand_landmarks[0]
        pixel_landmarks: List[Point] = []
        for lm in landmark_list:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_landmarks.append((px, py))

        handedness = "Right"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name or "Right"

        if draw:
            self._draw_task_landmarks(frame_bgr, pixel_landmarks)

        return {"landmarks": pixel_landmarks, "handedness": handedness}

    def process(self, frame_bgr, draw: bool = True) -> Optional[Dict[str, object]]:
        """
        Process frame and return:
        {
          "landmarks": List[(x, y)],  # 21 pixel points
          "handedness": "Left" | "Right"
        }
        """
        if frame_bgr is None:
            return None
        if self._backend == "legacy":
            return self._process_legacy(frame_bgr, draw=draw)
        return self._process_tasks(frame_bgr, draw=draw)
