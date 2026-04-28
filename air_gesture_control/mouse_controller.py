"""Mouse and scroll control layer using pyautogui."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import pyautogui

try:
    from .utils import clamp
except ImportError:
    from utils import clamp


class MouseController:
    """Controls cursor movement, click, drag, and scroll actions."""

    def __init__(
        self,
        smoothing: float = 0.35,
        sensitivity: float = 1.0,
        camera_margin_px: int = 80,
        click_cooldown_s: float = 0.35,
        scroll_sensitivity: float = 1.0,
        scroll_velocity_threshold: float = 80.0,
        scroll_smoothing_window: int = 5,
        scroll_min_sustain_frames: int = 3,
        scroll_debug: bool = False,
    ) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = clamp(smoothing, 0.01, 1.0)
        self.sensitivity = max(0.2, sensitivity)
        self.camera_margin_px = max(0, camera_margin_px)
        self.click_cooldown_s = max(0.05, click_cooldown_s)
        self.scroll_sensitivity = max(0.1, scroll_sensitivity)
        self.scroll_velocity_threshold = max(1.0, scroll_velocity_threshold)
        self.scroll_smoothing_window = max(1, int(scroll_smoothing_window))
        self.scroll_min_sustain_frames = max(1, int(scroll_min_sustain_frames))
        self.scroll_debug = scroll_debug

        self._prev_x = self.screen_w / 2
        self._prev_y = self.screen_h / 2
        self._last_click_time = 0.0
        self._dragging = False
        self._scroll_ref_y: Optional[int] = None
        self._scroll_prev_time: Optional[float] = None
        self._scroll_vel_buffer = deque(maxlen=self.scroll_smoothing_window)
        self._scroll_sustain_count = 0

    def _camera_to_screen(self, x: int, y: int, frame_w: int, frame_h: int) -> Tuple[float, float]:
        if frame_w <= 0 or frame_h <= 0:
            return self._prev_x, self._prev_y

        margin_x = min(self.camera_margin_px, frame_w // 3)
        margin_y = min(self.camera_margin_px, frame_h // 3)

        usable_w = max(1, frame_w - (2 * margin_x))
        usable_h = max(1, frame_h - (2 * margin_y))

        norm_x = clamp((x - margin_x) / usable_w, 0.0, 1.0)
        norm_y = clamp((y - margin_y) / usable_h, 0.0, 1.0)

        target_x = norm_x * (self.screen_w - 1)
        target_y = norm_y * (self.screen_h - 1)

        # Sensitivity scales displacement around the center.
        cx = (self.screen_w - 1) / 2.0
        cy = (self.screen_h - 1) / 2.0
        target_x = cx + (target_x - cx) * self.sensitivity
        target_y = cy + (target_y - cy) * self.sensitivity

        target_x = clamp(target_x, 1, self.screen_w - 2)
        target_y = clamp(target_y, 1, self.screen_h - 2)
        return target_x, target_y

    def move(self, x: int, y: int, frame_w: int, frame_h: int) -> None:
        target_x, target_y = self._camera_to_screen(x, y, frame_w, frame_h)
        smooth_x = self._prev_x + (target_x - self._prev_x) * self.smoothing
        smooth_y = self._prev_y + (target_y - self._prev_y) * self.smoothing

        self._prev_x = smooth_x
        self._prev_y = smooth_y

        pyautogui.moveTo(int(smooth_x), int(smooth_y))

    def click_if_ready(self) -> bool:
        now = time.perf_counter()
        if (now - self._last_click_time) < self.click_cooldown_s:
            return False
        pyautogui.click()
        self._last_click_time = now
        return True

    def double_click_if_ready(self) -> bool:
        now = time.perf_counter()
        if (now - self._last_click_time) < self.click_cooldown_s:
            return False
        pyautogui.doubleClick()
        self._last_click_time = now
        return True

    def start_drag(self) -> None:
        if not self._dragging:
            pyautogui.mouseDown()
            self._dragging = True

    def stop_drag(self) -> None:
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    def drag_move(self, x: int, y: int, frame_w: int, frame_h: int) -> None:
        self.start_drag()
        self.move(x, y, frame_w, frame_h)

    def reset_scroll_anchor(self) -> None:
        self._scroll_ref_y = None
        self._scroll_prev_time = None
        self._scroll_vel_buffer.clear()
        self._scroll_sustain_count = 0

    def scroll_from_hand(self, hand_y: int) -> None:
        if self._scroll_ref_y is None:
            self._scroll_ref_y = hand_y
            self._scroll_prev_time = time.perf_counter()
            return

        now = time.perf_counter()
        if self._scroll_prev_time is None:
            self._scroll_prev_time = now
            self._scroll_ref_y = hand_y
            return

        delta_t = now - self._scroll_prev_time
        if delta_t <= 0:
            self._scroll_prev_time = now
            self._scroll_ref_y = hand_y
            return

        # velocity in px/sec based on vertical movement.
        velocity = (hand_y - self._scroll_ref_y) / delta_t
        self._scroll_prev_time = now
        self._scroll_ref_y = hand_y

        self._scroll_vel_buffer.append(velocity)
        smooth_velocity = sum(self._scroll_vel_buffer) / len(self._scroll_vel_buffer)

        if abs(smooth_velocity) < self.scroll_velocity_threshold:
            self._scroll_sustain_count = 0
            if self.scroll_debug:
                print(f"[SCROLL] vel={velocity:.2f} smooth={smooth_velocity:.2f} (deadzone)")
            return

        self._scroll_sustain_count += 1
        if self._scroll_sustain_count < self.scroll_min_sustain_frames:
            if self.scroll_debug:
                print(
                    f"[SCROLL] vel={velocity:.2f} smooth={smooth_velocity:.2f} "
                    f"(sustain {self._scroll_sustain_count}/{self.scroll_min_sustain_frames})"
                )
            return

        # Negative because upward hand motion (decreasing y) should scroll up.
        amount = int(-smooth_velocity * 0.04 * self.scroll_sensitivity)
        amount = int(clamp(amount, -120, 120))
        if amount != 0:
            pyautogui.scroll(amount)
            if self.scroll_debug:
                print(f"[SCROLL] vel={velocity:.2f} smooth={smooth_velocity:.2f} amount={amount}")
