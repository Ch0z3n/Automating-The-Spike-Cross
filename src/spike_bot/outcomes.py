"""Conservative HUD observations; missing evidence never means success."""
from dataclasses import dataclass

import cv2
import numpy as np


def visible_lives(frame: np.ndarray) -> int | None:
    h, w = frame.shape[:2]
    present = []
    for center in (363, 414, 466, 518, 569):
        x, y = round(w * center / 1920), round(h * 134 / 1080)
        rx, ry = max(2, round(w * 12 / 1920)), max(2, round(h * 12 / 1080))
        patch = frame[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 160)
        warm = (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100)
        present.append(np.mean(white) > 0.15 and np.mean(warm) > 0.025)
    count = int(sum(present))
    # Icons disappear from the right. A gap is likely an occlusion, not a count.
    if not count or present != [True] * count + [False] * (5-count):
        return None
    return count


@dataclass
class LifeCounter:
    stable: int | None = None
    pending: int | None = None
    repeats: int = 0
    losses: int = 0

    def observe(self, value: int | None) -> int:
        if value is None:
            self.pending, self.repeats = None, 0
            return 0
        self.repeats = self.repeats + 1 if value == self.pending else 1
        self.pending = value
        if self.repeats < 3:
            return 0
        if self.stable is None:
            self.stable = value
            return 0
        # An increase requires an explicit round reset, never a silent one.
        if value >= self.stable:
            return 0
        lost = self.stable - value
        self.stable = value
        self.losses += lost
        return lost
