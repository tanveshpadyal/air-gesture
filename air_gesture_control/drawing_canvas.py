"""Air drawing canvas utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


class DrawingCanvas:
    """Maintains a persistent drawing layer over the camera frame."""

    def __init__(self, color=(0, 255, 0), thickness: int = 4) -> None:
        self.enabled = False
        self.color = color
        self.thickness = thickness
        self._canvas: Optional[np.ndarray] = None
        self._last_point: Optional[Point] = None

    def ensure_size(self, frame_shape) -> None:
        h, w = frame_shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._last_point = None

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        self._last_point = None
        return self.enabled

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas[:] = 0
        self._last_point = None

    def draw(self, point: Optional[Point]) -> None:
        if not self.enabled or self._canvas is None or point is None:
            self._last_point = None
            return

        if self._last_point is None:
            self._last_point = point
            return

        cv2.line(self._canvas, self._last_point, point, self.color, self.thickness)
        self._last_point = point

    def overlay_on(self, frame: np.ndarray) -> np.ndarray:
        if self._canvas is None:
            return frame

        gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self._canvas, self._canvas, mask=mask)
        return cv2.add(frame_bg, canvas_fg)
