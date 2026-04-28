"""Utility helpers for the air gesture control application."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Iterable, Optional, Tuple


Point = Tuple[int, int]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance between two 2D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


class FPSCounter:
    """Simple FPS tracker using wall clock deltas."""

    def __init__(self) -> None:
        self._prev = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        delta = now - self._prev
        self._prev = now
        if delta > 0:
            self._fps = 1.0 / delta
        return self._fps


class GestureStabilizer:
    """
    Returns a stabilized gesture only when the same gesture appears
    for `window_size` consecutive frames.
    """

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = max(1, window_size)
        self.history: Deque[str] = deque(maxlen=self.window_size)
        self.last_stable: str = "NONE"

    def reset(self) -> None:
        self.history.clear()
        self.last_stable = "NONE"

    def update(self, gesture: str) -> str:
        self.history.append(gesture)
        if len(self.history) < self.window_size:
            return self.last_stable

        first = self.history[0]
        if all(item == first for item in self.history):
            self.last_stable = first
        return self.last_stable


def average_points(points: Iterable[Point]) -> Optional[Point]:
    """Return average point from iterable, or None if empty."""
    pts = list(points)
    if not pts:
        return None
    x_sum = sum(p[0] for p in pts)
    y_sum = sum(p[1] for p in pts)
    return int(x_sum / len(pts)), int(y_sum / len(pts))
