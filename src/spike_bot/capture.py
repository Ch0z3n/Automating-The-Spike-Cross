from __future__ import annotations

from PIL import ImageGrab
import numpy as np


def grab_rgb(rect: dict[str, int]) -> np.ndarray:
    left, top = rect["left"], rect["top"]
    bbox = (left, top, left + rect["width"], top + rect["height"])
    return np.asarray(ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB"))


def sample_screen_rgb(x: int, y: int) -> list[int]:
    pixel = ImageGrab.grab(bbox=(x, y, x + 1, y + 1), all_screens=True).convert("RGB")
    return list(pixel.getpixel((0, 0)))

