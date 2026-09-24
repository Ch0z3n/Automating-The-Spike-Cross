from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time

import numpy as np
import cv2


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Observation:
    timestamp: float
    ball: Point | None
    player: Point | None
    ball_velocity: Point | None


def colour_centroid(
    frame: np.ndarray, target_rgb: list[int] | None, tolerance: float, min_pixels: int
) -> Point | None:
    if target_rgb is None:
        return None
    # int32 avoids overflow when squaring an RGB difference of up to 255.
    pixels = frame.astype(np.int32)
    target = np.asarray(target_rgb, dtype=np.int32)
    distance_squared = np.sum((pixels - target) ** 2, axis=2)
    ys, xs = np.nonzero(distance_squared <= tolerance * tolerance)
    if len(xs) < min_pixels:
        return None
    # Median is more robust than mean when the same colour appears in effects.
    return Point(float(np.median(xs)), float(np.median(ys)))


class Tracker:
    def __init__(self, history_size: int = 6) -> None:
        self._ball_history: deque[tuple[float, Point]] = deque(maxlen=history_size)
        self._previous_frame: np.ndarray | None = None
        self._misses = 0
        self._stationary_frames = 0
        self._excluded_circles: list[Point] = []

    def observe(
        self, frame: np.ndarray, vision: dict, timestamp: float | None = None
    ) -> Observation:
        now = time.monotonic() if timestamp is None else timestamp
        if vision.get("detector") == "screen":
            player = player_marker(frame)
            previous = self._ball_history[-1][1] if self._ball_history else None
            predicted = None
            velocity = self._velocity()
            if previous is not None and velocity is not None:
                dt = now - self._ball_history[-1][0]
                if 0 < dt <= 0.25:
                    predicted = Point(previous.x + velocity.x * dt, previous.y + velocity.y * dt)
            ball = detect_ball(
                frame,
                previous,
                player,
                vision.get("ball_roi"),
                self._excluded_circles,
                vision.get("ball_excluded_points"),
                predicted=predicted,
                previous_frame=self._previous_frame,
            )
            self._previous_frame = frame.copy()
        else:
            ball = colour_centroid(
                frame, vision["ball_rgb"], vision["ball_tolerance"], vision["ball_min_pixels"]
            )
            player = colour_centroid(
                frame, vision["player_rgb"], vision["player_tolerance"], vision["player_min_pixels"]
            )
        if ball is not None:
            if self._ball_history and np.hypot(
                ball.x - self._ball_history[-1][1].x,
                ball.y - self._ball_history[-1][1].y,
            ) < 2.0:
                self._stationary_frames += 1
            else:
                self._stationary_frames = 0
            if self._stationary_frames >= 4:
                self._excluded_circles.append(ball)
                self._excluded_circles = self._excluded_circles[-12:]
                self._ball_history.clear()
                self._stationary_frames = 0
                ball = None
        if ball is not None:
            if self._ball_history and now - self._ball_history[-1][0] > 0.25:
                self._ball_history.clear()
            self._ball_history.append((now, ball))
            self._misses = 0
        else:
            self._misses += 1
            if self._misses >= 5:
                self._ball_history.clear()
        velocity = self._velocity()
        return Observation(now, ball, player, velocity)

    def _velocity(self) -> Point | None:
        if len(self._ball_history) < 2:
            return None
        # Recent samples respond faster and avoid averaging separate parts of
        # the toss parabola into a misleading direction.
        history = list(self._ball_history)[-3:]
        t0, p0 = history[0]
        t1, p1 = self._ball_history[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return None
        return Point((p1.x - p0.x) / elapsed, (p1.y - p0.y) / elapsed)


def player_marker(frame_rgb: np.ndarray) -> Point | None:
    """Find the cyan triangle that identifies the controlled player."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (85, 100, 150), (115, 255, 255))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, Point]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        # The marker pulses from roughly 18x11 to 44x24 at 1080p.
        # Its large phase exceeds 500 pixels; rejecting that phase repeatedly
        # blinds both player tracking and ball detection during every toss.
        if (
            30 <= area <= 1000
            and 1.2 <= width / max(height, 1) <= 2.5
            and 0.25 <= area / max(width * height, 1) <= 0.80
            and width <= 60
            and height <= 35
            and y < frame_rgb.shape[0] * 0.62
        ):
            cx, cy = centroids[index]
            candidates.append((area, Point(float(cx), float(cy))))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _circle_candidates(frame_rgb: np.ndarray) -> list[tuple[float, float, float]]:
    play_height = int(frame_rgb.shape[0] * 0.78)
    gray = cv2.cvtColor(frame_rgb[:play_height], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=24,
        param1=110,
        param2=24,
        minRadius=8,
        maxRadius=28,
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    accepted = []
    for raw_x, raw_y, raw_radius in circles[0]:
        x, y, radius = (float(raw_x), float(raw_y), float(raw_radius))
        x0, x1 = max(0, int(x - radius)), min(frame_rgb.shape[1], int(x + radius + 1))
        y0, y1 = max(0, int(y - radius)), min(frame_rgb.shape[0], int(y + radius + 1))
        sample = hsv[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
        saturation, value, hue = sample[:, :, 1][inside], sample[:, :, 2][inside], sample[:, :, 0][inside]
        white = float(np.mean((saturation < 80) & (value > 175)))
        orange = float(np.mean((hue < 30) & (saturation > 100) & (value > 100)))
        # The volleyball has both a white body and orange/yellow curved bands.
        # This rejects the orange target ring, white court markings and HUD.
        if white > 0.18 and orange > 0.025:
            accepted.append((x, y, radius))
    return accepted


def detect_ball(
    frame_rgb: np.ndarray,
    previous: Point | None,
    player: Point | None,
    roi: list[float] | None = None,
    excluded: list[Point] | None = None,
    fixed_excluded: list[list[float]] | None = None,
    predicted: Point | None = None,
    previous_frame: np.ndarray | None = None,
) -> Point | None:
    """Track the white volleyball while rejecting the smaller target ring."""
    if player is None:
        return None  # Also prevents detections on menus and result screens.
    height, width = frame_rgb.shape[:2]
    left, right, top, bottom = roi or [0.04, 0.96, 0.15, 0.78]
    # Only pre-contact tracking is needed. Restricting the lower court avoids
    # character/court circles after the ball overlaps the player.
    bottom = min(bottom, 0.55)
    scored: list[tuple[float, Point]] = []
    for x, y, radius in _circle_candidates(frame_rgb):
        near_life_icon = any(np.hypot(x - width * icon_x, y - height * (134 / 1080)) < width * (20 / 1920)
                             for icon_x in (363 / 1920, 414 / 1920, 466 / 1920, 518 / 1920, 569 / 1920))
        if near_life_icon:
            # Prediction alone can lock onto a stationary icon. Require actual
            # changed pixels in its center as well as a recent moving track.
            if (predicted is None or previous is None or previous_frame is None
                    or previous_frame.shape != frame_rgb.shape
                    or np.hypot(x - predicted.x, y - predicted.y) > 25
                    or np.hypot(x - previous.x, y - previous.y) < 4):
                continue
            size = max(3, int(radius * 0.5))
            x0, x1 = max(0, int(x)-size), min(width, int(x)+size+1)
            y0, y1 = max(0, int(y)-size), min(height, int(y)+size+1)
            difference = np.max(np.abs(frame_rgb[y0:y1, x0:x1].astype(np.int16)
                                      - previous_frame[y0:y1, x0:x1].astype(np.int16)), axis=2)
            if difference.size == 0 or np.mean(difference > 30) < 0.15:
                continue
        if not (width * left <= x <= width * right and height * top <= y <= height * bottom):
            continue  # scoreboard balls and HUD
        if any(np.hypot(x - point.x, y - point.y) < 35 for point in (excluded or [])):
            continue
        if any(
            np.hypot(x - width * point[0], y - height * point[1]) < 42
            for point in (fixed_excluded or [])
        ):
            continue
        score = abs(radius - 23.0) * 5.0
        if previous is not None:
            distance = np.hypot(x - previous.x, y - previous.y)
            score += distance
            if distance > 260:
                score += 500
        else:
            score += np.hypot(x - player.x, y - player.y) * 0.35
        scored.append((score, Point(x, y)))
    return min(scored, default=(0, None), key=lambda item: item[0])[1]


def detect_moving_ball(
    frame_rgb: np.ndarray,
    previous_frame_rgb: np.ndarray | None,
    previous: Point | None,
    player: Point | None,
    roi: list[float] | None = None,
) -> Point | None:
    """Find the moving volleyball while static drill targets disappear."""
    if previous_frame_rgb is None or player is None:
        return None
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    old_gray = cv2.cvtColor(previous_frame_rgb, cv2.COLOR_RGB2GRAY)
    difference = cv2.absdiff(gray, old_gray)
    motion = cv2.threshold(difference, 24, 255, cv2.THRESH_BINARY)[1]
    motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, (0, 0, 165), (179, 100, 255))
    warm = cv2.inRange(hsv, (0, 80, 100), (40, 255, 255))
    ball_colours = cv2.dilate(cv2.bitwise_or(white, warm), np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(motion, ball_colours)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, Point]] = []
    height, width = frame_rgb.shape[:2]
    left, right, top, bottom = roi or [0.04, 0.96, 0.15, 0.78]
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if not (
            10 <= area <= 1100
            and 2 <= box_width <= 65
            and 2 <= box_height <= 95
            and width * left <= x <= width * right
            and height * top <= y <= height * bottom
        ):
            continue
        cx, cy = (float(value) for value in centroids[index])
        point = Point(cx, cy)
        if np.hypot(cx - player.x, cy - player.y) < 28:
            continue  # moving pixels belonging to the cyan marker itself
        # A ball component is compact. Thin court lines and large player
        # animations receive penalties, while temporal continuity dominates.
        compact_penalty = abs(box_width - box_height) * 1.5 + max(area - 500, 0) * 0.08
        if previous is not None:
            continuity = np.hypot(cx - previous.x, cy - previous.y)
            if continuity > 180:
                continuity += 350
        else:
            continuity = np.hypot(cx - player.x, cy - player.y) * 0.35
        candidates.append((float(continuity + compact_penalty), point))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]
