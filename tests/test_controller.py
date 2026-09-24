import unittest
from unittest.mock import patch

from spike_bot.controller import SpikeBrain, decide, KeyboardController, Decision
from spike_bot.vision import Observation, Point


SETTINGS = {
    "target_lead_seconds": 0.2,
    "position_deadzone_pixels": 30,
    "attack_height_ratio": 0.55,
    "ball_gravity_pixels_per_second2": 1580.0,
    "contact_height_ratio": 0.40,
    "jump_lead_seconds": 0.72,
    "jump_lead_tolerance_seconds": 0.08,
    "jump_trigger_max_height_ratio": 0.28,
    "fallback_jump_min_height_ratio": 0.17,
    "fallback_jump_max_height_ratio": 0.26,
    "fallback_max_velocity_y": 80.0,
    "peak_spike_delay_seconds": 0.50,
    "angled_jump_lead_seconds": 0.68,
    "angled_spike_delay_seconds": 0.50,
    "angled_target_offset_pixels": 70,
    "angled_velocity_threshold": 55,
    "peak_player_height_ratio": 0.40,
    "action_hold_seconds": 0.08,
    "dynamic_target_min_x_ratio": 0.08,
    "dynamic_target_max_x_ratio": 0.52,
}


class DecisionTests(unittest.TestCase):
    def test_airborne_intercept_does_not_follow_outgoing_ball(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.22})
        brain.phase = "airborne"
        brain.jump_time = 1.0
        brain.target_x = 500
        brain.decide(Observation(1.1, Point(650, 190), Point(500, 300), Point(600, 100)), (1080, 1920, 3))
        self.assertEqual(brain.target_x, 500)

    def test_left_arc_gap_keeps_intercept_and_attempts_jump(self):
        brain = SpikeBrain({**SETTINGS, "fixed_target_x_ratio": 0.365, "action_cooldown_seconds": 0.22})
        first = brain.decide(Observation(1.0, Point(590, 167), Point(503, 541), Point(-113, -318)), (1080, 1920, 3))
        self.assertFalse(first.action)
        gap = brain.decide(Observation(1.1, None, Point(503, 541), None), (1080, 1920, 3))
        self.assertTrue(gap.action)
        self.assertIn("predicted-through-gap", gap.reason)
        self.assertNotEqual(gap.movement, "right")

    def test_fine_correction_releases_continuous_movement(self):
        keyboard = KeyboardController({"left": "LEFT", "right": "RIGHT", "action": "Z"}, SETTINGS)
        with patch.object(keyboard, "_set_key") as key, patch.object(keyboard, "_tap") as tap:
            keyboard.apply(Decision("right", False, "approach"))
            keyboard.apply(Decision("left", False, "fine correction", 0.04))
            self.assertIsNone(keyboard._movement)
            key.assert_called_with("RIGHT", False)
            tap.assert_called_once_with("LEFT", 0.04)

    def test_near_lane_jitter_does_not_restart_movement(self):
        brain = SpikeBrain({**SETTINGS, "position_deadzone_pixels": 45})
        self.assertEqual(brain.approach(260), "right")
        brain.player_velocity_x = 980
        brain.observation_time = 0.2
        self.assertIsNone(brain.approach(154))
        brain.observation_time = 0.5
        brain.player_velocity_x = 0
        for delta in (20, 30, -25, -30):
            self.assertIsNone(brain.approach(delta))
        movement = brain.approach(-52)
        self.assertEqual(movement, "left")
        pulse = brain.movement_pulse(-52, movement)
        self.assertAlmostEqual(pulse, 32 / 900)
        self.assertIsNone(brain.approach(-52))

    def test_brakes_before_crossing_lane(self):
        brain = SpikeBrain(SETTINGS)
        brain.player_velocity_x = 800
        self.assertIsNone(brain.approach(90))
        self.assertEqual(brain.approach(250), "right")
        brain = SpikeBrain(SETTINGS)
        brain.player_velocity_x = -800
        self.assertIsNone(brain.approach(-90))
        self.assertEqual(brain.approach(-250), "left")

    def test_movement_holds_without_blocking_and_releases_on_stop(self):
        keyboard = KeyboardController({"left": "LEFT", "right": "RIGHT", "action": "Z"}, SETTINGS)
        with patch.object(keyboard, "_set_key") as key, patch("spike_bot.controller.time.sleep") as sleep:
            keyboard.apply(Decision("right", False, "approach"))
            keyboard.apply(Decision("right", False, "approach"))
            keyboard.apply(Decision("left", False, "correct"))
            keyboard.release()
            self.assertEqual([call.args for call in key.call_args_list],
                [("RIGHT", True), ("RIGHT", False), ("LEFT", True), ("LEFT", False)])
            sleep.assert_not_called()

    def test_waits_without_tracking(self):
        result = decide(Observation(0, None, None, None), (720, 1280, 3), SETTINGS)
        self.assertIsNone(result.movement)
        self.assertFalse(result.action)

    def test_moves_toward_projected_ball(self):
        observation = Observation(0, Point(500, 200), Point(300, 600), Point(100, 50))
        result = decide(observation, (720, 1280, 3), SETTINGS)
        self.assertEqual(result.movement, "right")

    def test_attacks_when_descending_low_and_aligned(self):
        observation = Observation(0, Point(310, 500), Point(300, 600), Point(0, 120))
        result = decide(observation, (720, 1280, 3), SETTINGS)
        self.assertTrue(result.action)

    def test_spike_brain_jumps_at_calibrated_rising_eta(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        observation = Observation(1.0, Point(310, 150), Point(300, 600), Point(0, -370))
        result = brain.decide(observation, (720, 1280, 3))
        self.assertTrue(result.action)
        self.assertIn("jump", result.reason)

    def test_spike_brain_does_not_jump_before_reaching_lane(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        observation = Observation(1.0, Point(390, 150), Point(300, 600), Point(0, -370))
        result = brain.decide(observation, (720, 1280, 3))
        self.assertFalse(result.action)
        self.assertEqual(result.movement, "right")

    def test_spike_brain_uses_fixed_arc_when_velocity_is_missing(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        brain.last_measured_ball_time = 0.7
        brain.last_measured_velocity_y = -450
        observation = Observation(1.0, Point(310, 180), Point(300, 600), None)
        result = brain.decide(observation, (720, 1280, 3))
        self.assertTrue(result.action)
        self.assertIn("jump", result.reason)

    def test_spike_brain_rejects_late_reappearing_ball(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        brain.last_measured_ball_time = 0.3
        brain.last_measured_velocity_y = -450
        observation = Observation(1.0, Point(310, 180), Point(300, 600), None)
        result = brain.decide(observation, (720, 1280, 3))
        self.assertFalse(result.action)

    def test_visible_angled_toss_overrides_waiting_lane(self):
        brain = SpikeBrain({**SETTINGS, "fixed_target_x_ratio": 0.10, "action_cooldown_seconds": 0.2})
        observation = Observation(1.0, Point(500, 150), Point(400, 600), Point(100, -370))
        result = brain.decide(observation, (720, 1280, 3))
        self.assertEqual(result.movement, "right")
        self.assertEqual(brain.trajectory, "angled")

    def test_spike_brain_spikes_when_airborne_and_close(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        brain.phase = "airborne"
        brain.jump_time = 1.0
        brain.target_x = 310
        observation = Observation(2.0, Point(310, 270), Point(300, 220), Point(0, 50))
        result = brain.decide(observation, (720, 1280, 3))
        self.assertTrue(result.action)
        self.assertIn("spike", result.reason)

    def test_spike_timer_survives_visual_occlusion(self):
        brain = SpikeBrain({**SETTINGS, "action_cooldown_seconds": 0.2})
        brain.phase = "airborne"
        brain.jump_time = 1.0
        brain.last_action = 1.0
        result = brain.decide(Observation(1.5, None, None, None), (720, 1280, 3))
        self.assertTrue(result.action)
        self.assertEqual(result.reason, "spike: calibrated jump timer")

    def test_prepositions_to_fixed_lane_before_ball_appears(self):
        brain = SpikeBrain({**SETTINGS, "fixed_target_x_ratio": 0.365, "action_cooldown_seconds": 0.2})
        result = brain.decide(Observation(1.0, None, Point(400, 600), None), (720, 1280, 3))
        self.assertEqual(result.movement, "right")
        self.assertFalse(result.action)


if __name__ == "__main__":
    unittest.main()
