from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
from datetime import datetime
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from PIL import Image

from .capture import grab_rgb, sample_screen_rgb
from .autotune import AutoTuner
from .config import load_config, save_config
from .controller import Decision, KeyboardController, SpikeBrain
from .vision import Tracker


HOTKEYS = {"ball": 0x77, "player": 0x78, "arm": 0x79, "quit": 0x7B}  # F8/F9/F10/F12


def pressed(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 1)


def cursor_position() -> tuple[int, int]:
    point = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The Spike Cross training bot")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--armed", action="store_true", help="Start with keyboard control armed")
    parser.add_argument("--autotune", action="store_true", help="Run rounds, restart, and search timings automatically")
    return parser.parse_args()


def ensure_config(path: Path) -> None:
    if not path.exists():
        example = Path("config.example.json")
        if not example.exists():
            raise FileNotFoundError("config.json and config.example.json are missing")
        shutil.copyfile(example, path)
        print(f"Created {path}; adjust the capture rectangle, then run again.")
        raise SystemExit(0)


def main() -> int:
    ctypes.windll.user32.SetProcessDPIAware()
    args = parse_args()
    ensure_config(args.config)
    config = load_config(args.config)
    tracker = Tracker()
    keyboard = KeyboardController(config["keys"], config["controller"])
    brain = SpikeBrain(config["controller"])
    tuner = AutoTuner(config["controller"], config["capture"], Path("autotune_state.json")) if args.autotune else None
    armed = args.armed or args.autotune
    last_report = 0.0
    interval = 1.0 / max(args.fps, 1.0)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    frame_dir = log_path.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    recent_frames: deque[tuple[int, object]] = deque(maxlen=8)
    debug_remaining = 0
    frame_number = 0
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="debug-images")
    pending_images = deque()

    def save_frame(number, pixels):
        # Bounded queue: a slow disk must never hold up game input or exhaust RAM.
        while pending_images and pending_images[0].done():
            pending_images.popleft().result()
        if len(pending_images) < 24:
            pending_images.append(writer.submit(
                Image.fromarray(pixels).save,
                frame_dir / f"frame-{number:06d}.jpg", quality=82,
            ))
    print("F8 sample ball | F9 sample player | F10 arm/disarm | F12 quit")
    print(f"Event log: {log_path.resolve()}")

    try:
      while True:
        started = time.monotonic()
        if pressed(HOTKEYS["quit"]):
            break
        for label, field in (("ball", "ball_rgb"), ("player", "player_rgb")):
            if pressed(HOTKEYS[label]):
                x, y = cursor_position()
                config["vision"][field] = sample_screen_rgb(x, y)
                save_config(args.config, config)
                print(f"Sampled {label}: RGB {config['vision'][field]}")
        if pressed(HOTKEYS["arm"]):
            armed = not armed
            if not armed:
                keyboard.release()
            print("ARMED - sending keys" if armed else "DISARMED - observation only")

        frame = grab_rgb(config["capture"])
        observation = tracker.observe(frame, config["vision"])
        decision = brain.decide(observation, frame.shape)
        tune_event = tuner.observe(frame, observation, decision) if tuner else None
        if tuner and tuner.controls_suspended:
            decision = Decision(None, False, "autotune UI transition")
        if tune_event:
            print(tune_event)
            keyboard.release()
            click_mode = tuner.click_mode
            # The first toss begins almost immediately after Start Game.  The
            # old generic transition delay slept through it, so start clicks
            # must return to vision tracking straight away.
            time.sleep(0.08 if click_mode == "start" else 1.0)
            tuner.restart()
            brain = SpikeBrain(config["controller"])
            tracker = Tracker()
            time.sleep(0.12 if click_mode == "start" else 1.5)
        input_started = time.monotonic()
        if armed:
            keyboard.apply(decision)
        event = {
            "frame": frame_number,
            "input_time": input_started,
            "observation_to_input_ms": round((input_started - observation.timestamp) * 1000, 2),
            "ui_event": tune_event,
            "visible_lives": tuner.last_visible_lives if tuner else None,
            "stable_lives": tuner.life_counter.stable if tuner else None,
            "time": observation.timestamp,
            "armed": armed,
            "ball": None if observation.ball is None else [observation.ball.x, observation.ball.y],
            "player": None if observation.player is None else [observation.player.x, observation.player.y],
            "velocity": None if observation.ball_velocity is None else [observation.ball_velocity.x, observation.ball_velocity.y],
            "movement": decision.movement,
            "movement_pulse_seconds": decision.movement_pulse_seconds,
            "action": decision.action,
            "reason": decision.reason,
        }
        log.write(json.dumps(event, separators=(",", ":")) + "\n")
        if decision.action:
            log.flush()
            for old_number, old_frame in recent_frames:
                save_frame(old_number, old_frame)
            debug_remaining = 10
        if debug_remaining > 0 or frame_number % 5 == 0 or tune_event:
            save_frame(frame_number, frame.copy())
            debug_remaining = max(0, debug_remaining - 1)
        recent_frames.append((frame_number, frame.copy()))
        frame_number += 1
        if started - last_report >= 0.5:
            ball = "missing" if observation.ball is None else f"({observation.ball.x:.0f},{observation.ball.y:.0f})"
            player = "missing" if observation.player is None else f"({observation.player.x:.0f},{observation.player.y:.0f})"
            mode = "ARMED" if armed else "watching"
            print(f"[{mode}] ball={ball} player={player} move={decision.movement} action={decision.action} | {decision.reason}")
            last_report = started
        time.sleep(max(0.0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.release()
        log.close()
        writer.shutdown(wait=True)
    print("Stopped; keyboard control is off.")
    return 0
