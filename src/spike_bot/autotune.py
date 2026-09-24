from __future__ import annotations

import ctypes
from dataclasses import dataclass, asdict
import json
import shutil
from math import hypot
from pathlib import Path
import time

import cv2
import numpy as np

from .controller import Decision, time_to_height
from .vision import Observation
from .outcomes import LifeCounter, visible_lives


@dataclass
class Trial:
    jump_time: float
    spike_time: float | None = None
    jump_registered: bool = False
    closest_distance: float = 9999.0
    max_post_spike_vx: float = 0.0
    ball_y_at_jump: float | None = None
    vx_at_jump: float | None = None
    predicted_contact_x: float | None = None
    trajectory: str = "unknown"
    outcome: str = "unknown"
    likely_contact: bool = False
    post_spike_deflection_frames: int = 0

    @property
    def contact_score(self) -> float:
        geometry = max(0.0, 1.0 - self.closest_distance / 150.0)
        deflection = min(1.0, self.max_post_spike_vx / 220.0)
        return (0.25 if self.jump_registered else 0.0) + geometry * 0.5 + deflection * 0.25


class AutoTuner:
    """Small, persistent parameter search for the repeatable Spike Drill."""

    def __init__(self, settings: dict, capture: dict, state_path: Path) -> None:
        self.settings = settings
        self.capture = capture
        self.state_path = state_path
        self.candidates = self._candidates()
        self.state = self._load()
        self._recompute_history()
        self.trials: list[Trial] = []
        self.current: Trial | None = None
        self.last_player_y: float | None = None
        self.result_seen_at: float | None = None
        self.start_seen_at: float | None = None
        self.start_click_at = 0.0
        self.click_mode = "restart"
        self.error_click_at = 0.0
        self.error_seen_at: float | None = None
        self.error_clicks = 0
        self.recovering_error = False
        self.error_cleared_at: float | None = None
        self.controls_suspended = False
        self.idle_missing_at: float | None = None
        self.training_click_at = 0.0
        self.life_counter = LifeCounter()
        self.last_visible_lives: int | None = None
        self.unassigned_life_losses = 0
        self.apply_candidate()

    @staticmethod
    def _candidates() -> list[dict[str, float]]:
        # Sample-efficient local search around the manually established region.
        return [
            {"jump_lead_seconds": lead, "peak_spike_delay_seconds": delay, "fixed_target_x_ratio": lane}
            for lead, delay, lane in (
                (0.72, 0.50, 0.365),
                (0.76, 0.50, 0.365),
                (0.68, 0.50, 0.365),
                (0.72, 0.52, 0.365),
                (0.72, 0.48, 0.365),
                (0.74, 0.50, 0.365),
                (0.70, 0.50, 0.365),
                (0.76, 0.52, 0.365),
                (0.68, 0.48, 0.365),
            )
        ]

    def _load(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") == 2:
                return state
            # Old data split timer-driven spikes into spurious new jump trials.
            # Preserve it for analysis, but never rank new candidates against it.
            backup = self.state_path.with_name(self.state_path.stem + f".legacy-{time.time_ns()}.json")
            shutil.copy2(self.state_path, backup)
        return {"schema_version": 2, "candidate_index": 0, "round": 0, "best_score": -1.0, "best": None, "history": []}

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    @staticmethod
    def _trial_score(trial: dict) -> float:
        distance = float(trial.get("closest_distance", 9999.0))
        geometry = max(0.0, 1.0 - distance / 150.0)
        return (0.35 if trial.get("jump_registered") else 0.0) + geometry * 0.65

    def _recompute_history(self) -> None:
        best_fitness = -1.0
        best = None
        for entry in self.state.get("history", []):
            trials = entry.get("trials", [])
            count = len(trials)
            total = sum(self._trial_score(trial) for trial in trials)
            per_trial = total / count if count else 0.0
            registered = sum(bool(trial.get("jump_registered")) for trial in trials)
            # Do not crown a tiny, lucky round over a well-sampled one.
            fitness = per_trial * min(1.0, count / 10.0)
            entry["score"] = round(total, 3)
            entry["score_per_trial"] = round(per_trial, 3)
            entry["registered_rate"] = round(registered / count, 3) if count else 0.0
            entry["fitness"] = round(fitness, 3)
            if fitness > best_fitness:
                best_fitness = fitness
                best = entry.get("candidate")
        self.state["best_score"] = round(best_fitness, 3)
        self.state["best"] = best
        if self.state.get("history"):
            self._save()

    def apply_candidate(self) -> None:
        candidate = self.candidates[self.state["candidate_index"] % len(self.candidates)]
        self.settings.update(candidate)

    @staticmethod
    def is_result_screen(frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        # The Steam window may be letterboxed; the banner occupies the middle
        # band of the captured monitor rather than the nominal game viewport.
        roi = frame[int(h * 0.36):int(h * 0.62), int(w * 0.30):int(w * 0.76)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(hsv, np.array([24, 120, 190]), np.array([42, 255, 255]))
        count, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
        glyphs = sum(1 for area in stats[1:count, cv2.CC_STAT_AREA] if 500 <= area <= 6000)
        return glyphs >= 8 and cv2.countNonZero(yellow) > 12000

    @staticmethod
    def is_start_screen(frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        panel = frame[int(h * 0.12):int(h * 0.66), int(w * 0.80):int(w * 0.99)]
        if float(cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY).mean()) > 65:
            return False
        roi = frame[int(h * 0.74):int(h * 0.92), int(w * 0.70):int(w * 0.94)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        orange = cv2.inRange(hsv, np.array([5, 170, 210]), np.array([25, 255, 255]))
        return cv2.countNonZero(orange) > 5000

    @staticmethod
    def is_client_error(frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.28):int(h * 0.67), int(w * 0.31):int(w * 0.69)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        # START! is bright orange on a darkened court. Only neutral white
        # dialog lettering counts as error text, never the orange intro banner.
        white_text = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 150)
        return float(gray.mean()) < 45 and float(np.mean(gray < 45)) > 0.70 and int(np.sum(white_text)) > 400

    def observe(self, frame: np.ndarray, observation: Observation, decision: Decision) -> str | None:
        now = observation.timestamp
        player, ball, velocity = observation.player, observation.ball, observation.ball_velocity
        self.controls_suspended = False
        plausible_player = player is not None and player.y >= frame.shape[0] * 0.30
        if plausible_player and self.recovering_error:
            # A real on-court marker is definitive evidence that gameplay has
            # resumed; never let stale menu recovery suppress a live toss.
            self.recovering_error = False
            self.error_clicks = 0
            self.error_cleared_at = None
        if self.is_client_error(frame) and not plausible_player:
            self.controls_suspended = True
            # Countdown digits also resemble white text on a dark panel.
            # A real modal persists; the complete 3-2-1 intro does not.
            if self.error_seen_at is None:
                self.error_seen_at = now
            if now - self.error_seen_at < 4.0:
                return None
            self.recovering_error = True
            self.error_cleared_at = None
            if self.error_clicks < 2 and time.monotonic() - self.error_click_at > 1.5:
                self.error_click_at = time.monotonic()
                self.click_mode = "error_inner" if self.error_clicks == 0 else "error_outer"
                self.error_clicks += 1
                return f"AUTOTUNE Client Error detected; dismissing dialog {self.error_clicks}/2"
            return None
        self.error_seen_at = None
        if self.recovering_error:
            self.controls_suspended = True
            if self.error_cleared_at is None:
                self.error_cleared_at = time.monotonic()
            elif time.monotonic() - self.error_cleared_at > 1.0:
                self.recovering_error = False
                self.error_clicks = 0
                self.click_mode = "training"
                return "AUTOTUNE error cleared; reopening Spike Drill"
        if decision.action and decision.reason.startswith("jump:"):
            if self.current is not None:
                self.trials.append(self.current)
            predicted_x = None
            vx = None if velocity is None else velocity.x
            if ball is not None:
                predicted_x = ball.x
                if velocity is not None:
                    eta = time_to_height(
                        ball.y,
                        velocity.y,
                        frame.shape[0] * self.settings["contact_height_ratio"],
                        self.settings["ball_gravity_pixels_per_second2"],
                    )
                    if eta is not None:
                        predicted_x += velocity.x * eta
            waiting_x = frame.shape[1] * self.settings.get("fixed_target_x_ratio", 0.365)
            angled = (vx is not None and abs(vx) >= 55) or (
                predicted_x is not None and abs(predicted_x - waiting_x) >= 70
            )
            self.current = Trial(
                jump_time=now,
                ball_y_at_jump=None if ball is None else ball.y,
                vx_at_jump=vx,
                predicted_contact_x=predicted_x,
                trajectory="angled" if angled else "normal",
            )
        elif decision.action and decision.reason.startswith("spike:") and self.current is not None:
            self.current.spike_time = now

        self.last_visible_lives = visible_lives(frame) if plausible_player else None
        lost = self.life_counter.observe(self.last_visible_lives)
        if lost:
            if self.current is not None and now - self.current.jump_time < 2.0 and self.current.outcome == "unknown":
                self.current.outcome = "life_lost_after_attempt"
                self.unassigned_life_losses += lost - 1
            else:
                self.unassigned_life_losses += lost

        if self.current is not None:
            if player is not None and player.y < frame.shape[0] * 0.46:
                self.current.jump_registered = True
            if player is not None and ball is not None:
                self.current.closest_distance = min(
                    self.current.closest_distance, hypot(ball.x - player.x, ball.y - player.y)
                )
            if self.current.spike_time is not None and velocity is not None and now - self.current.spike_time < 0.9:
                self.current.max_post_spike_vx = max(self.current.max_post_spike_vx, abs(velocity.x))
                baseline = self.current.vx_at_jump
                changed = (ball is not None and baseline is not None
                           and abs(velocity.x - baseline) > 300
                           and abs(velocity.x) > 300)
                self.current.post_spike_deflection_frames = self.current.post_spike_deflection_frames + 1 if changed else 0
                if self.current.post_spike_deflection_frames >= 2:
                    self.current.likely_contact = True

        if self.is_result_screen(frame):
            self.controls_suspended = True
            if self.result_seen_at is None:
                self.result_seen_at = time.monotonic()
                self.click_mode = "restart"
                if self.current is None and not self.trials:
                    return "AUTOTUNE found existing result screen; restarting"
                return self.finish_round()
            if time.monotonic() - self.result_seen_at > 3.0:
                self.result_seen_at = time.monotonic()
                self.click_mode = "restart"
                return "AUTOTUNE result screen still present; retrying restart"
        else:
            self.result_seen_at = None
        if self.is_start_screen(frame) and player is None:
            self.controls_suspended = True
            self.idle_missing_at = None
            if self.start_seen_at is None:
                self.start_seen_at = time.monotonic()
            elif time.monotonic() - self.start_seen_at > 0.8 and time.monotonic() - self.start_click_at > 3.0:
                self.start_click_at = time.monotonic()
                self.click_mode = "start"
                return "AUTOTUNE player screen; clicking Start Game"
            return None
        else:
            self.start_seen_at = None
        # Missing detections are not evidence of a menu: countdowns, loading,
        # and occlusion all hide both objects. Only recognized UI may click.
        return None

    def finish_round(self) -> str:
        if self.current is not None:
            self.trials.append(self.current)
            self.current = None
        score = sum(self._trial_score(asdict(trial)) for trial in self.trials)
        count = len(self.trials)
        score_per_trial = score / count if count else 0.0
        fitness = score_per_trial * min(1.0, count / 10.0)
        candidate = dict(self.candidates[self.state["candidate_index"] % len(self.candidates)])
        entry = {
            "outcome_observations": {
                "version": 1,
                "observed_life_losses": self.life_counter.losses,
                "unassigned_life_losses": self.unassigned_life_losses,
                "likely_contacts": sum(t.likely_contact for t in self.trials),
                "confirmed_in_bounds_hits": None,
            },
            "round": self.state["round"],
            "candidate": candidate,
            "score": round(score, 3),
            "score_per_trial": round(score_per_trial, 3),
            "fitness": round(fitness, 3),
            "trials": [asdict(t) | {"contact_score": round(t.contact_score, 3)} for t in self.trials],
        }
        entry["trajectory_breakdown"] = {
            kind: {
                "trials": len(group),
                "score_per_trial": round(
                    sum(self._trial_score(asdict(trial)) for trial in group) / len(group), 3
                ) if group else 0.0,
            }
            for kind in ("normal", "angled", "unknown")
            if (group := [trial for trial in self.trials if trial.trajectory == kind])
        }
        self.state["history"].append(entry)
        if fitness > self.state["best_score"]:
            self.state["best_score"] = round(fitness, 3)
            self.state["best"] = candidate
        self.state["round"] += 1
        self.state["candidate_index"] = (self.state["candidate_index"] + 1) % len(self.candidates)
        self._save()
        self.trials = []
        self.life_counter = LifeCounter()
        self.unassigned_life_losses = 0
        self.apply_candidate()
        return f"AUTOTUNE round complete: proxy={score:.2f}, next={self.settings}"

    def restart(self) -> None:
        # ImageGrab reports physical pixels. Opt out of DPI virtualization so
        # SetCursorPos uses the same coordinate space on scaled displays.
        ctypes.windll.user32.SetProcessDPIAware()
        if self.click_mode in ("error_inner", "error_outer"):
            y_ratio = 0.62 if self.click_mode == "error_inner" else 0.56
            x = int(self.capture["left"] + self.capture["width"] * 0.50)
            y = int(self.capture["top"] + self.capture["height"] * y_ratio)
            ctypes.windll.user32.SetCursorPos(x, y)
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            time.sleep(0.08)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            time.sleep(0.15)
            ctypes.windll.user32.keybd_event(0x0D, 0, 0, 0)
            time.sleep(0.10)
            ctypes.windll.user32.keybd_event(0x0D, 0, 0x0002, 0)
            return
        if self.click_mode == "start":
            x_ratio, y_ratio = 0.80, 0.90
        elif self.click_mode == "training":
            x_ratio, y_ratio = 0.339, 0.440
        else:
            # Training Complete restart icon.
            x_ratio, y_ratio = 0.055, 0.925
        x = int(self.capture["left"] + self.capture["width"] * x_ratio)
        y = int(self.capture["top"] + self.capture["height"] * y_ratio)
        ctypes.windll.user32.SetCursorPos(x, y)
        repeats = 2 if self.click_mode == "training" else 1
        hold = 0.18 if self.click_mode == "training" else 0.06
        for _ in range(repeats):
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            time.sleep(hold)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            time.sleep(0.18)
