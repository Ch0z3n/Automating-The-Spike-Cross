from __future__ import annotations

from dataclasses import dataclass
import ctypes
import time

from .vision import Observation, Point


VK = {
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "SPACE": 0x20,
    "X": 0x58,
    "Z": 0x5A,
    "C": 0x43,
}
KEYEVENTF_KEYUP = 0x0002


@dataclass(frozen=True)
class Decision:
    movement: str | None
    action: bool
    reason: str
    movement_pulse_seconds: float | None = None


class SpikeBrain:
    """Two-stage manual-mode controller: approach/jump, then spike."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.phase = "grounded"
        self.last_action = 0.0
        self.grounded_frames = 0
        self.jump_time: float | None = None
        self.target_x: float | None = None
        self.trajectory = "normal"
        self.last_measured_ball_time: float | None = None
        self.last_measured_velocity_y: float | None = None
        self.last_player_sample: tuple[float, float] | None = None
        self.player_velocity_x = 0.0
        self.approach_direction: str | None = None
        self.approach_settle_until = 0.0
        self.observation_time = 0.0
        self.last_ball_sample: Observation | None = None

    def approach(self, delta: float) -> str | None:
        deadzone = self.settings["position_deadzone_pixels"]
        now = self.observation_time
        if now < self.approach_settle_until:
            return None
        # Once aligned, small prediction/marker jitter must not restart a run.
        restart_distance = self.settings.get("movement_restart_pixels", 35.0)
        if self.approach_direction is None and abs(delta) <= restart_distance:
            return None
        if self.approach_direction is None and abs(self.player_velocity_x) < 150 and abs(delta) <= 120:
            return "right" if delta > 0 else "left"
        # Account for capture/input delay and the player's momentum after release.
        future_delta = delta - self.player_velocity_x * self.settings.get("braking_lead_seconds", 0.12)
        if abs(delta) <= deadzone or abs(future_delta) <= deadzone or delta * future_delta < 0:
            if self.approach_direction is not None:
                self.approach_settle_until = now + self.settings.get("movement_settle_seconds", 0.22)
            self.approach_direction = None
            return None
        direction = "right" if delta > 0 else "left"
        if self.approach_direction is not None and direction != self.approach_direction:
            self.approach_direction = None
            self.approach_settle_until = now + self.settings.get("movement_settle_seconds", 0.22)
            return None
        self.approach_direction = direction
        return direction

    def movement_pulse(self, delta: float, movement: str | None) -> float | None:
        if movement is not None and self.approach_direction is None:
            # At measured running speed (~900 px/s), a 40 ms tap corrects
            # roughly 36 px without holding a key through a slow vision frame.
            self.approach_settle_until = self.observation_time + 0.3
            return min(0.06, max(0.025, (abs(delta) - 20.0) / 900.0))
        return None

    def decide(self, observation: Observation, frame_shape: tuple[int, ...]) -> Decision:
        ball, player = observation.ball, observation.player
        now = observation.timestamp
        self.observation_time = now
        predicted_ball = False
        if ball is not None:
            self.last_ball_sample = observation
        elif self.last_ball_sample is not None:
            sample = self.last_ball_sample
            dt = now - sample.timestamp
            if sample.ball_velocity is not None and 0 < dt <= 0.35:
                gravity = self.settings["ball_gravity_pixels_per_second2"]
                ball = Point(sample.ball.x + sample.ball_velocity.x * dt,
                             sample.ball.y + sample.ball_velocity.y * dt + 0.5 * gravity * dt * dt)
                observation = Observation(now, ball, player, Point(sample.ball_velocity.x, sample.ball_velocity.y + gravity * dt))
                predicted_ball = True
        if player is not None:
            self.player_velocity_x = 0.0
            if self.last_player_sample is not None:
                previous_time, previous_x = self.last_player_sample
                elapsed = now - previous_time
                if 0.02 <= elapsed <= 0.3:
                    self.player_velocity_x = max(-1400.0, min(1400.0, (player.x - previous_x) / elapsed))
            self.last_player_sample = (now, player.x)
        peak_delay = (
            self.settings["angled_spike_delay_seconds"]
            if self.trajectory == "angled"
            else self.settings["peak_spike_delay_seconds"]
        )

        # Once a jump has started, do not let the brief contact-time occlusion
        # suppress the spike key. The calibrated second press is clock-driven.
        if (
            self.phase in ("jumping", "airborne")
            and self.jump_time is not None
            and now - self.jump_time >= peak_delay
            and now - self.last_action >= self.settings["action_cooldown_seconds"]
        ):
            self.phase = "spiked"
            self.last_action = now
            return Decision(None, True, "spike: calibrated jump timer")

        if player is None:
            return Decision(None, False, "waiting for stable tracking")

        height = frame_shape[0]
        airborne = player.y < height * 0.46
        if airborne:
            self.grounded_frames = 0
            if self.phase in ("grounded", "jumping"):
                self.phase = "airborne"
        else:
            self.grounded_frames += 1
            if self.grounded_frames >= 4 and self.phase in ("airborne", "spiked"):
                self.phase = "grounded"
                self.jump_time = None
                self.target_x = None
                self.trajectory = "normal"

        # If overlap hides the ball, retain a conservative late-contact fallback
        # rather than abandoning the spike phase entirely.
        if ball is None:
            fallback = (
                self.phase == "airborne"
                and self.jump_time is not None
                and now - self.jump_time >= peak_delay
                and player.y < height * 0.40
            )
            if fallback:
                self.phase = "spiked"
                self.last_action = now
                return Decision(None, True, "spike: visual-overlap fallback")
            fixed_target = self.settings.get("fixed_target_x_ratio")
            if self.phase == "grounded" and fixed_target is not None:
                retaining = (self.target_x is not None and self.last_ball_sample is not None
                             and now - self.last_ball_sample.timestamp <= 0.9)
                destination = self.target_x if retaining else frame_shape[1] * fixed_target
                delta = destination - player.x
                deadzone = self.settings["position_deadzone_pixels"]
                movement = self.approach(delta)
                return Decision(movement, False, f"pre-position: dx={delta:+.0f}px", self.movement_pulse(delta, movement))
            return Decision(None, False, "waiting for ball; retaining phase")

        velocity = observation.ball_velocity
        estimated_velocity_y = None if velocity is None else velocity.y
        if (
            estimated_velocity_y is None
            and self.last_measured_ball_time is not None
            and self.last_measured_velocity_y is not None
        ):
            elapsed = now - self.last_measured_ball_time
            estimated_velocity_y = self.last_measured_velocity_y + (
                self.settings["ball_gravity_pixels_per_second2"] * elapsed
            )
        contact_time = None
        if velocity is not None:
            contact_time = time_to_height(
                ball.y,
                velocity.y,
                height * self.settings["contact_height_ratio"],
                self.settings["ball_gravity_pixels_per_second2"],
            )
        # High-altitude observations are clean; near contact the ball overlaps
        # the player and motion segmentation can latch onto an arm or shirt.
        # Predict each toss's horizontal crossing point while the ball is
        # high, then retain the smoothed target through brief tracking gaps.
        if self.phase == "grounded" and ball.y < frame_shape[0] * 0.42:
            predicted_x = ball.x
            if velocity is not None and contact_time is not None:
                predicted_x += velocity.x * contact_time
            left = frame_shape[1] * self.settings["dynamic_target_min_x_ratio"]
            right = frame_shape[1] * self.settings["dynamic_target_max_x_ratio"]
            predicted_x = min(max(predicted_x, left), right)
            waiting_x = frame_shape[1] * self.settings.get("fixed_target_x_ratio", 0.365)
            if abs(predicted_x - waiting_x) >= self.settings["angled_target_offset_pixels"] or (
                velocity is not None and abs(velocity.x) >= self.settings["angled_velocity_threshold"]
            ):
                self.trajectory = "angled"
            self.target_x = predicted_x if self.target_x is None else self.target_x * 0.55 + predicted_x * 0.45
        projected_x = self.target_x if self.target_x is not None else ball.x
        delta = projected_x - player.x
        deadzone = self.settings["position_deadzone_pixels"]
        movement = self.approach(delta)

        cooldown_ready = now - self.last_action >= self.settings["action_cooldown_seconds"]
        lead = (
            self.settings["angled_jump_lead_seconds"]
            if self.trajectory == "angled"
            else self.settings["jump_lead_seconds"]
        )
        lead_tolerance = self.settings["jump_lead_tolerance_seconds"]
        predictive_jump = (
            contact_time is not None
            and velocity is not None
            and velocity.y < 0
            and ball.y < height * self.settings["jump_trigger_max_height_ratio"]
            and lead - lead_tolerance <= contact_time <= lead + lead_tolerance
        )
        fixed_arc_fallback = (
            contact_time is None
            and estimated_velocity_y is not None
            and estimated_velocity_y <= self.settings.get("fallback_max_velocity_y", 80.0)
            and height * self.settings["fallback_jump_min_height_ratio"] <= ball.y
            <= height * self.settings["fallback_jump_max_height_ratio"]
        )
        ball_near_hitting_hand = (
            self.jump_time is not None
            and now - self.jump_time >= 0.12
            and 20 < ball.y - player.y < 120
            and player.y < height * self.settings["peak_player_height_ratio"]
            and abs(projected_x - player.x) < deadzone * 1.7
        )
        peak_timed_contact = (
            self.jump_time is not None
            and now - self.jump_time >= peak_delay
            and player.y < height * self.settings["peak_player_height_ratio"]
            and abs(projected_x - player.x) < deadzone * 1.7
        )
        action = False
        verb = "track"
        if (
            self.phase == "grounded"
            and (predictive_jump or fixed_arc_fallback)
            and abs(delta) < deadzone * 1.25
            and cooldown_ready
        ):
            action = True
            verb = "jump"
            self.phase = "jumping"
            self.last_action = now
            self.jump_time = now
        elif (
            self.phase == "airborne"
            and airborne
            and (ball_near_hitting_hand or peak_timed_contact)
            and cooldown_ready
        ):
            action = True
            verb = "spike"
            self.last_action = now
            self.phase = "spiked"
        eta = "?" if contact_time is None else f"{contact_time:.2f}s"
        reason = f"{verb}: dx={delta:+.0f}px ball=({ball.x:.0f},{ball.y:.0f}) eta={eta} trajectory={self.trajectory} phase={self.phase}"
        if predicted_ball:
            reason += " predicted-through-gap"
        if velocity is not None:
            self.last_measured_ball_time = now
            self.last_measured_velocity_y = velocity.y
        # Do not delay the jump key with a fine-positioning tap.
        if action:
            movement = None
        return Decision(movement, action, reason, self.movement_pulse(delta, movement))


def time_to_height(y: float, velocity_y: float, target_y: float, gravity: float) -> float | None:
    """Solve the ball parabola for its next crossing of target_y."""
    # 0.5*g*t^2 + velocity*t + (y-target) = 0
    discriminant = velocity_y * velocity_y - 2.0 * gravity * (y - target_y)
    if gravity <= 0 or discriminant < 0:
        return None
    root = (-velocity_y + discriminant ** 0.5) / gravity
    return root if root >= 0 else None


def decide(observation: Observation, frame_shape: tuple[int, ...], settings: dict) -> Decision:
    ball, player = observation.ball, observation.player
    if ball is None or player is None:
        return Decision(None, False, "waiting for stable tracking")

    lead = settings["target_lead_seconds"]
    projected_x = ball.x
    if observation.ball_velocity is not None:
        projected_x += observation.ball_velocity.x * lead

    delta = projected_x - player.x
    deadzone = settings["position_deadzone_pixels"]
    movement = "right" if delta > deadzone else "left" if delta < -deadzone else None

    height_ratio = ball.y / frame_shape[0]
    descending = observation.ball_velocity is not None and observation.ball_velocity.y > 0
    aligned = abs(delta) <= deadzone * 1.6
    action = height_ratio >= settings["attack_height_ratio"] and descending and aligned
    reason = f"projected dx={delta:+.0f}px, ball height={height_ratio:.2f}"
    return Decision(movement, action, reason)


class KeyboardController:
    def __init__(self, keys: dict[str, str], settings: dict) -> None:
        self.keys = keys
        self.settings = settings
        self.last_action = 0.0
        self._movement: str | None = None

    def release(self) -> None:
        if self._movement is not None:
            self._set_key(self.keys[self._movement], False)
            self._movement = None

    @staticmethod
    def _set_key(name: str, down: bool) -> None:
        vk = VK.get(name.upper())
        if vk is None:
            if len(name) != 1:
                raise ValueError(f"Unsupported key: {name}")
            vk = ord(name.upper())
        ctypes.windll.user32.keybd_event(vk, 0, 0 if down else KEYEVENTF_KEYUP, 0)

    @staticmethod
    def _tap(name: str, hold: float) -> None:
        vk = VK.get(name.upper())
        if vk is None:
            if len(name) != 1:
                raise ValueError(f"Unsupported key: {name}")
            vk = ord(name.upper())
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        try:
            time.sleep(hold)
        finally:
            ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

    def apply(self, decision: Decision) -> None:
        if decision.movement is not None and decision.movement_pulse_seconds is not None:
            self.release()
            self._tap(self.keys[decision.movement], decision.movement_pulse_seconds)
            return
        if decision.movement != self._movement:
            self.release()
            if decision.movement:
                self._set_key(self.keys[decision.movement], True)
                self._movement = decision.movement
        now = time.monotonic()
        if decision.action and now - self.last_action >= self.settings["action_cooldown_seconds"]:
            self._tap(self.keys["action"], self.settings["action_hold_seconds"])
            self.last_action = now
