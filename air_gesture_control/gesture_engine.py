"""Priority-based gesture resolution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from .utils import GestureStabilizer, clamp, euclidean_distance
except ImportError:
    from utils import GestureStabilizer, clamp, euclidean_distance


Point = Tuple[int, int]

THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_MCP = 17
PINKY_TIP = 20
PINKY_PIP = 18


@dataclass
class GestureContext:
    landmarks: List[Point]
    handedness: str
    finger_status: List[int]
    index_middle_distance: float
    thumb_index_distance: float
    click_distance_px: float
    thumb_index_click_distance_px: float


@dataclass
class GestureResolution:
    finger_status: List[int]
    raw_gesture: str
    stable_gesture: str
    confidence_score: float
    priority: int
    pinch_distance: float


class Gesture:
    """Base gesture interface."""

    name: str = "NONE"
    priority: int = 0

    def __init__(self) -> None:
        self.confidence_score: float = 0.0

    def _set(self, active: bool, confidence: float) -> bool:
        self.confidence_score = clamp(confidence, 0.0, 1.0) if active else 0.0
        return active

    def is_active(self, context: GestureContext) -> bool:
        raise NotImplementedError


class DragGesture(Gesture):
    name = "DRAG"
    priority = 5

    def is_active(self, context: GestureContext) -> bool:
        thumb, index, middle, ring, pinky = context.finger_status
        active = index == 0 and middle == 0 and ring == 0 and pinky == 0
        confidence = 1.0 if thumb == 0 else 0.85
        return self._set(active, confidence)


class ClickGesture(Gesture):
    name = "CLICK"
    priority = 4

    def is_active(self, context: GestureContext) -> bool:
        _, index, middle, ring, pinky = context.finger_status
        index_mcp = context.landmarks[INDEX_MCP]
        pinky_mcp = context.landmarks[PINKY_MCP]
        palm_width = max(1.0, euclidean_distance(index_mcp, pinky_mcp))

        # Strict click: index+middle pinch only (do not trigger on fingers-up alone).
        pinch_threshold = min(
            context.click_distance_px * 0.65,
            max(18.0, 0.26 * palm_width),
        )
        idx_tip = context.landmarks[INDEX_TIP]
        mid_tip = context.landmarks[MIDDLE_TIP]
        dx = abs(idx_tip[0] - mid_tip[0])
        dy = abs(idx_tip[1] - mid_tip[1])

        active = (
            index == 1
            and middle == 1
            and ring == 0
            and pinky == 0
            and context.index_middle_distance < pinch_threshold
            and dx < (pinch_threshold * 1.05)
            and dy < (pinch_threshold * 1.35)
        )
        confidence = 1.0 - (
            context.index_middle_distance / max(1.0, pinch_threshold)
        )
        return self._set(active, confidence)


class DoubleClickGesture(Gesture):
    name = "DOUBLE_CLICK"
    priority = 6

    def is_active(self, context: GestureContext) -> bool:
        _, index, middle, ring, pinky = context.finger_status
        # Double-click gesture: index + thumb pinch, others down.
        thumb_tip = context.landmarks[THUMB_TIP]
        index_mcp = context.landmarks[INDEX_MCP]
        pinky_mcp = context.landmarks[PINKY_MCP]

        palm_width = max(1.0, euclidean_distance(index_mcp, pinky_mcp))
        pinch_threshold = max(
            20.0,
            min(context.thumb_index_click_distance_px * 0.80, 0.35 * palm_width),
        )
        idx_tip = context.landmarks[INDEX_TIP]
        dx = abs(thumb_tip[0] - idx_tip[0])
        dy = abs(thumb_tip[1] - idx_tip[1])

        active = (
            index == 1
            and middle == 0
            and ring == 0
            and pinky == 0
            and context.thumb_index_distance < pinch_threshold
            and dx < (pinch_threshold * 1.15)
            and dy < (pinch_threshold * 1.45)
        )
        confidence = 1.0 - (
            context.thumb_index_distance / max(1.0, pinch_threshold)
        )
        return self._set(active, confidence)


class ScrollGesture(Gesture):
    name = "SCROLL"
    priority = 3

    def is_active(self, context: GestureContext) -> bool:
        thumb, index, middle, ring, pinky = context.finger_status
        active = index == 1 and middle == 1 and ring == 1 and pinky == 1
        confidence = 0.8 + (0.2 if thumb == 1 else 0.0)
        return self._set(active, confidence)


class MoveGesture(Gesture):
    name = "MOVE"
    priority = 1

    def is_active(self, context: GestureContext) -> bool:
        _, index, middle, ring, pinky = context.finger_status
        active = index == 1 and middle == 0 and ring == 0 and pinky == 0
        # Lower confidence when fingers are very close, because that indicates a click.
        click_proximity = 1.0 - clamp(
            context.thumb_index_distance / max(1.0, context.thumb_index_click_distance_px),
            0.0,
            1.0,
        )
        confidence = 0.95 - (0.45 * click_proximity)
        return self._set(active, confidence)


def get_finger_status(
    landmarks: List[Point],
    handedness: str = "Right",
    tolerance_px: int = 8,
) -> List[int]:
    """
    Returns [thumb, index, middle, ring, pinky] where 1=up and 0=down.
    """
    if len(landmarks) != 21:
        return [0, 0, 0, 0, 0]

    thumb_tip_x = landmarks[THUMB_TIP][0]
    thumb_ip_x = landmarks[THUMB_IP][0]

    if handedness == "Right":
        thumb_up = 1 if thumb_tip_x < (thumb_ip_x + tolerance_px) else 0
    else:
        thumb_up = 1 if thumb_tip_x > (thumb_ip_x - tolerance_px) else 0

    index_up = 1 if landmarks[INDEX_TIP][1] < (landmarks[INDEX_PIP][1] + tolerance_px) else 0
    middle_up = 1 if landmarks[MIDDLE_TIP][1] < (landmarks[MIDDLE_PIP][1] + tolerance_px) else 0
    ring_up = 1 if landmarks[RING_TIP][1] < (landmarks[RING_PIP][1] + tolerance_px) else 0
    pinky_up = 1 if landmarks[PINKY_TIP][1] < (landmarks[PINKY_PIP][1] + tolerance_px) else 0

    return [thumb_up, index_up, middle_up, ring_up, pinky_up]


class GestureEngine:
    """
    Priority-based gesture resolver.
    Active gestures are sorted by:
    1) Highest priority
    2) Highest confidence score
    """

    def __init__(
        self,
        click_distance_px: float = 48.0,
        stabilize_frames: int = 2,
        finger_tolerance_px: int = 8,
        thumb_index_click_distance_px: float = 52.0,
        log_conflicts: bool = True,
    ) -> None:
        self.click_distance_px = click_distance_px
        self.finger_tolerance_px = max(0, finger_tolerance_px)
        self.thumb_index_click_distance_px = thumb_index_click_distance_px
        self.log_conflicts = log_conflicts
        self.stabilizer = GestureStabilizer(window_size=stabilize_frames)
        self._last_conflict_signature = ""
        self._gestures: List[Gesture] = [
            DragGesture(),
            DoubleClickGesture(),
            ClickGesture(),
            ScrollGesture(),
            MoveGesture(),
        ]

    def reset(self) -> None:
        self.stabilizer.reset()
        self._last_conflict_signature = ""

    def detect(self, landmarks: List[Point], handedness: str = "Right") -> GestureResolution:
        if len(landmarks) != 21:
            return GestureResolution(
                finger_status=[0, 0, 0, 0, 0],
                raw_gesture="NONE",
                stable_gesture=self.stabilizer.last_stable,
                confidence_score=0.0,
                priority=0,
                pinch_distance=0.0,
            )

        finger_status = get_finger_status(
            landmarks,
            handedness=handedness,
            tolerance_px=self.finger_tolerance_px,
        )
        index_middle_distance = euclidean_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
        thumb_index_distance = euclidean_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        context = GestureContext(
            landmarks=landmarks,
            handedness=handedness,
            finger_status=finger_status,
            index_middle_distance=index_middle_distance,
            thumb_index_distance=thumb_index_distance,
            click_distance_px=self.click_distance_px,
            thumb_index_click_distance_px=self.thumb_index_click_distance_px,
        )

        active_gestures: List[Gesture] = []
        for gesture in self._gestures:
            if gesture.is_active(context):
                active_gestures.append(gesture)

        if not active_gestures:
            self._last_conflict_signature = ""
            raw = "NONE"
            confidence = 0.0
            priority = 0
        else:
            active_gestures.sort(key=lambda g: (g.priority, g.confidence_score), reverse=True)
            selected = active_gestures[0]
            raw = selected.name
            confidence = selected.confidence_score
            priority = selected.priority

            if len(active_gestures) > 1 and self.log_conflicts:
                contender = active_gestures[1]
                signature = f"{selected.name}|{contender.name}"
                if signature != self._last_conflict_signature:
                    print(
                        f"Gesture conflict: {selected.name} vs {contender.name} -> {selected.name} selected"
                    )
                    self._last_conflict_signature = signature
            else:
                self._last_conflict_signature = ""

        stable = self.stabilizer.update(raw)
        return GestureResolution(
            finger_status=finger_status,
            raw_gesture=raw,
            stable_gesture=stable,
            confidence_score=confidence,
            priority=priority,
            pinch_distance=index_middle_distance,
        )
