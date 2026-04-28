"""Backward-compatible gesture exports.

This module preserves old import paths while delegating logic to gesture_engine.
"""

from __future__ import annotations

try:
    from .gesture_engine import GestureEngine as GestureDetector
    from .gesture_engine import GestureResolution as GestureResult
    from .gesture_engine import get_finger_status
except ImportError:
    from gesture_engine import GestureEngine as GestureDetector
    from gesture_engine import GestureResolution as GestureResult
    from gesture_engine import get_finger_status


__all__ = ["GestureDetector", "GestureResult", "get_finger_status"]
