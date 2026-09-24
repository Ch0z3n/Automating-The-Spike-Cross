import tempfile
import unittest
from pathlib import Path

import numpy as np

from spike_bot.autotune import AutoTuner
from spike_bot.controller import Decision
from spike_bot.vision import Observation, Point


class AutoTuneTests(unittest.TestCase):
    def test_fragmented_orange_intro_is_not_result_banner(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for index in range(8):
            x = 650 + index * 70
            frame[430:490, x:x + 40] = (200, 140, 0)
        self.assertFalse(AutoTuner.is_result_screen(frame))

    def test_missing_tracking_never_clicks_training_tile(self):
        with tempfile.TemporaryDirectory() as folder:
            tuner = AutoTuner({}, {}, Path(folder) / "state.json")
            tuner.idle_missing_at = 0.0
            frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
            for now in (10, 20, 30):
                event = tuner.observe(frame, Observation(now, None, None, None), Decision(None, False, "waiting"))
                self.assertIsNone(event)

    def test_transient_countdown_does_not_trigger_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            tuner = AutoTuner({}, {}, Path(folder) / "state.json")
            dark = np.full((1080, 1920, 3), 15, dtype=np.uint8)
            dark[500:540, 850:1070] = 230
            idle = Decision(None, False, "waiting")
            for now in (10, 11, 12, 13):
                self.assertIsNone(tuner.observe(dark, Observation(now, None, None, None), idle))
                self.assertFalse(tuner.recovering_error)
            clear = np.full_like(dark, 120)
            tuner.observe(clear, Observation(13.1, None, Point(700, 540), None), idle)
            self.assertIsNone(tuner.error_seen_at)
            tuner.observe(dark, Observation(20, None, None, None), idle)
            event = tuner.observe(dark, Observation(24.1, None, None, None), idle)
            self.assertIn("Client Error", event)

    def test_orange_start_banner_is_not_client_error(self):
        frame = np.full((1080, 1920, 3), 15, dtype=np.uint8)
        frame[470:640, 640:1280] = (255, 160, 0)
        self.assertFalse(AutoTuner.is_client_error(frame))

    def test_timer_spike_belongs_to_original_jump(self):
        with tempfile.TemporaryDirectory() as folder:
            tuner = AutoTuner({"ball_gravity_pixels_per_second2": 1580, "contact_height_ratio": 0.4}, {}, Path(folder) / "state.json")
            frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
            tuner.observe(frame, Observation(1, Point(700, 210), Point(700, 540), None), Decision(None, True, "jump: aligned"))
            tuner.observe(frame, Observation(1.5, Point(700, 300), Point(700, 320), None), Decision(None, True, "spike: calibrated jump timer"))
            self.assertEqual(tuner.current.jump_time, 1)
            self.assertEqual(tuner.current.spike_time, 1.5)
            self.assertEqual(tuner.trials, [])

    def test_detects_yellow_training_complete_banner(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for index in range(8):
            x = 650 + index * 70
            frame[430:490, x:x + 40] = (255, 210, 20)
        self.assertTrue(AutoTuner.is_result_screen(frame))

    def test_rejects_normal_dark_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertFalse(AutoTuner.is_result_screen(frame))

    def test_detects_orange_start_button(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[820:950, 1450:1750] = (255, 160, 15)
        self.assertTrue(AutoTuner.is_start_screen(frame))

    def test_detects_dark_client_error_dialog(self):
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        frame[300:720, 600:1320] = 15
        frame[500:540, 850:1070] = 230
        self.assertTrue(AutoTuner.is_client_error(frame))

    def test_persists_round_history(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = {}
            tuner = AutoTuner(settings, {"left": 0, "top": 0, "width": 1920, "height": 1080}, Path(folder) / "state.json")
            message = tuner.finish_round()
            self.assertIn("round complete", message)
            self.assertEqual(tuner.state["round"], 1)
            self.assertEqual(len(tuner.state["history"]), 1)


if __name__ == "__main__":
    unittest.main()
