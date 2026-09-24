"""Record low-rate game frames for offline calibration; sends no input."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import time

from PIL import ImageGrab


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("captures"))
    args = parser.parse_args()
    session = args.output / datetime.now().strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    interval = 1 / args.fps
    start = time.monotonic()
    frame = 0
    print(f"Recording to {session.resolve()}", flush=True)
    while time.monotonic() - start < args.seconds:
        tick = time.monotonic()
        image = ImageGrab.grab(all_screens=True)
        image.save(session / f"frame-{frame:05d}.jpg", quality=70)
        frame += 1
        time.sleep(max(0, interval - (time.monotonic() - tick)))
    print(f"Saved {frame} frames", flush=True)


if __name__ == "__main__":
    main()
