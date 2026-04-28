"""Entry point for the air gesture control application."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2

try:
    from .age_detector import AgeDetector
    from .drawing_canvas import DrawingCanvas
    from .gesture_engine import GestureEngine
    from .hand_tracker import HandTracker
    from .mouse_controller import MouseController
    from .utils import FPSCounter
except ImportError:
    from age_detector import AgeDetector
    from drawing_canvas import DrawingCanvas
    from gesture_engine import GestureEngine
    from hand_tracker import HandTracker
    from mouse_controller import MouseController
    from utils import FPSCounter


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    smoothing: float = 0.35
    sensitivity: float = 1.0
    camera_margin_px: int = 90
    click_cooldown_s: float = 0.35
    click_distance_px: float = 48.0
    thumb_index_click_distance_px: float = 52.0
    scroll_sensitivity: float = 1.0
    scroll_velocity_threshold: float = 80.0
    scroll_smoothing_window: int = 5
    scroll_debug: bool = False
    stabilize_frames: int = 2
    finger_tolerance_px: int = 8
    age_detection_enabled: bool = False
    age_inference_interval: int = 3
    show_landmarks: bool = True


class AirGestureApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tracker = HandTracker(max_num_hands=1)
        self.gesture_engine = GestureEngine(
            click_distance_px=config.click_distance_px,
            stabilize_frames=config.stabilize_frames,
            finger_tolerance_px=config.finger_tolerance_px,
            thumb_index_click_distance_px=config.thumb_index_click_distance_px,
        )
        self.mouse = MouseController(
            smoothing=config.smoothing,
            sensitivity=config.sensitivity,
            camera_margin_px=config.camera_margin_px,
            click_cooldown_s=config.click_cooldown_s,
            scroll_sensitivity=config.scroll_sensitivity,
            scroll_velocity_threshold=config.scroll_velocity_threshold,
            scroll_smoothing_window=config.scroll_smoothing_window,
            scroll_debug=config.scroll_debug,
        )
        self.canvas = DrawingCanvas()
        self.age_detector = AgeDetector(
            enabled=config.age_detection_enabled,
            inference_interval=config.age_inference_interval,
        )
        self.fps = FPSCounter()
        self._last_stable_gesture = "NONE"
        self._age_faces_count = 0

        self.cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions/device index.")

    def cleanup(self) -> None:
        self.mouse.stop_drag()
        self.tracker.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _draw_ui(self, frame, mode: str, gesture_text: str, fps_value: float) -> None:
        age_status = "ON" if self.age_detector.enabled else "OFF"
        age_info = f"Age: {age_status} | Faces: {self._age_faces_count}"
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Gesture: {gesture_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps_value:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            age_info,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 240, 255),
            2,
        )
        cv2.putText(
            frame,
            "ESC: Exit | D: Toggle Draw | C: Clear Draw | A: Toggle Age",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if self.age_detector.warning:
            cv2.putText(
                frame,
                f"Age Warning: {self.age_detector.warning}",
                (10, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 80, 255),
                1,
            )

    def run(self) -> None:
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            self.canvas.ensure_size(frame.shape)
            frame_h, frame_w = frame.shape[:2]

            gesture_text = "NONE"
            mode_text = "Draw" if self.canvas.enabled else "Mouse"

            hand_data = self.tracker.process(frame, draw=self.config.show_landmarks)
            if hand_data is None:
                self.gesture_engine.reset()
                self.mouse.stop_drag()
                self.mouse.reset_scroll_anchor()
                self.canvas.draw(None)
                self._last_stable_gesture = "NONE"
            else:
                landmarks = hand_data["landmarks"]
                handedness = hand_data["handedness"]
                result = self.gesture_engine.detect(landmarks, handedness=handedness)
                gesture_text = f"{result.stable_gesture} (raw:{result.raw_gesture})"

                index_tip = landmarks[8]

                if self.canvas.enabled:
                    # Draw only while index is the only raised finger.
                    finger = result.finger_status
                    if finger[1] == 1 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0:
                        self.canvas.draw(index_tip)
                    else:
                        self.canvas.draw(None)

                    self.mouse.stop_drag()
                    self.mouse.reset_scroll_anchor()
                    self._last_stable_gesture = "NONE"
                else:
                    stable = result.stable_gesture
                    if stable == "MOVE":
                        self.mouse.stop_drag()
                        self.mouse.reset_scroll_anchor()
                        self.mouse.move(index_tip[0], index_tip[1], frame_w, frame_h)
                    elif stable == "CLICK":
                        self.mouse.stop_drag()
                        self.mouse.reset_scroll_anchor()
                        if self._last_stable_gesture != "CLICK":
                            self.mouse.click_if_ready()
                    elif stable == "DOUBLE_CLICK":
                        self.mouse.stop_drag()
                        self.mouse.reset_scroll_anchor()
                        if self._last_stable_gesture != "DOUBLE_CLICK":
                            self.mouse.double_click_if_ready()
                    elif stable == "DRAG":
                        self.mouse.reset_scroll_anchor()
                        self.mouse.drag_move(index_tip[0], index_tip[1], frame_w, frame_h)
                    elif stable == "SCROLL":
                        self.mouse.stop_drag()
                        self.mouse.scroll_from_hand(landmarks[9][1])
                    else:
                        self.mouse.stop_drag()
                        self.mouse.reset_scroll_anchor()
                    self._last_stable_gesture = stable

            self._age_faces_count = self.age_detector.process_and_draw(frame)
            display = self.canvas.overlay_on(frame)
            fps_value = self.fps.update()
            self._draw_ui(display, mode_text, gesture_text, fps_value)
            cv2.imshow("Air Gesture Control", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord("d"), ord("D")):
                self.canvas.toggle()
            if key in (ord("c"), ord("C")):
                self.canvas.clear()
            if key in (ord("a"), ord("A")):
                self.age_detector.toggle()


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Air Gesture Control (OpenCV + MediaPipe)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--margin", type=int, default=90)
    parser.add_argument("--click-distance", type=float, default=48.0)
    parser.add_argument("--thumb-index-click-distance", type=float, default=52.0)
    parser.add_argument("--click-cooldown", type=float, default=0.35)
    parser.add_argument("--scroll-sensitivity", type=float, default=1.0)
    parser.add_argument("--scroll-velocity-threshold", type=float, default=80.0)
    parser.add_argument("--scroll-smoothing-window", type=int, default=5)
    parser.add_argument("--scroll-debug", action="store_true")
    parser.add_argument("--stabilize-frames", type=int, default=2)
    parser.add_argument("--finger-tolerance", type=int, default=8)
    parser.add_argument("--age-on", action="store_true")
    parser.add_argument("--age-interval", type=int, default=3)
    parser.add_argument("--hide-landmarks", action="store_true")
    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera_index,
        smoothing=args.smoothing,
        sensitivity=args.sensitivity,
        camera_margin_px=args.margin,
        click_distance_px=args.click_distance,
        thumb_index_click_distance_px=args.thumb_index_click_distance,
        click_cooldown_s=args.click_cooldown,
        scroll_sensitivity=args.scroll_sensitivity,
        scroll_velocity_threshold=args.scroll_velocity_threshold,
        scroll_smoothing_window=args.scroll_smoothing_window,
        scroll_debug=args.scroll_debug,
        stabilize_frames=args.stabilize_frames,
        finger_tolerance_px=args.finger_tolerance,
        age_detection_enabled=args.age_on,
        age_inference_interval=args.age_interval,
        show_landmarks=not args.hide_landmarks,
    )


def main() -> int:
    config = parse_args()
    app = AirGestureApp(config)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        app.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
