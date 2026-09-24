"""Evaluate visual tracking against a recorded drill session."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


GAME_CROP = (0, 148, 1920, 1080)


def crop_game(frame: np.ndarray) -> np.ndarray:
    x, y, width, height = GAME_CROP
    return frame[y : y + height, x : x + width]


def player_marker(frame: np.ndarray) -> tuple[float, float] | None:
    """Locate the cyan triangle above the controlled player."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 100, 150), (115, 255, 255))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 30 <= area <= 500 and 1.2 <= width / max(height, 1) <= 2.5 and width <= 45 and height <= 30 and y < 810:
            candidates.append((area, tuple(centroids[index])))
    return max(candidates, default=(0, None))[1]


def ball_candidates(frame: np.ndarray) -> list[tuple[float, float, float]]:
    """Return circle candidates in the playable portion of the court."""
    gray = cv2.cvtColor(frame[:850], cv2.COLOR_BGR2GRAY)
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
    return [tuple(float(value) for value in circle) for circle in circles[0]]


def choose_ball(
    candidates: list[tuple[float, float, float]],
    previous: tuple[float, float] | None,
    player: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if not candidates or player is None:
        return None
    scored = []
    for x, y, radius in candidates:
        # Ball radius is normally about 15 px at 1080p. Continuity is the
        # strongest signal after the first observation.
        score = abs(radius - 23) * 5
        if previous is not None:
            score += np.hypot(x - previous[0], y - previous[1])
        elif player is not None:
            distance = np.hypot(x - player[0], y - player[1])
            score += min(distance, 500) * 0.35
        # Exclude HUD and floor markings.
        if y < 80 or y > 780:
            score += 500
        scored.append((score, (x, y)))
    return min(scored)[1]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    previous = None
    found_ball = found_player = processed = 0
    for path in sorted(args.recording.glob("frame-*.jpg"))[:: args.stride]:
        source = cv2.imread(str(path))
        if source is None:
            continue
        frame = crop_game(source)
        player = player_marker(frame)
        ball = choose_ball(ball_candidates(frame), previous, player)
        processed += 1
        if player is not None:
            found_player += 1
        if ball is not None:
            found_ball += 1
            previous = ball
    print(f"frames={processed} player={found_player} ({found_player/max(processed,1):.1%}) ball={found_ball} ({found_ball/max(processed,1):.1%})")


if __name__ == "__main__":
    main()
