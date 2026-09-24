import unittest
from unittest.mock import patch

import numpy as np
import cv2

from spike_bot.vision import Point, Tracker, colour_centroid, detect_ball, detect_moving_ball, player_marker


class VisionTests(unittest.TestCase):
    def test_life_zone_requires_motion_and_matching_trajectory(self):
        old = np.zeros((1080, 1920, 3), dtype=np.uint8)
        old[124:145, 508:529] = (245, 200, 120)
        changed = old.copy()
        changed[124:145, 508:529] = (250, 250, 250)
        cases = [
            (old, Point(518, 134), None),
            (changed, Point(700, 134), None),
            (changed, Point(518, 134), Point(518, 134)),
        ]
        with patch("spike_bot.vision._circle_candidates", return_value=[(518, 134, 23)]):
            for frame, predicted, expected in cases:
                with self.subTest(predicted=predicted, expected=expected):
                    ball = detect_ball(frame, Point(540, 145), Point(503, 540),
                                       [0.08, 0.72, 0.085, 0.78],
                                       predicted=predicted, previous_frame=old)
                    self.assertEqual(ball, expected)

    def test_left_arc_above_lives_strip_is_visible(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("spike_bot.vision._circle_candidates", return_value=[(518, 134, 23), (540, 103, 23)]):
            ball = detect_ball(frame, None, Point(503, 540), [0.08, 0.72, 0.085, 0.78])
        self.assertEqual(ball, Point(540, 103))

    def test_high_toss_is_kept_but_life_icons_are_excluded(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("spike_bot.vision._circle_candidates", return_value=[(400, 134, 23), (689, 134, 23)]):
            ball = detect_ball(frame, None, Point(400, 540), [0.08, 0.72, 0.085, 0.78])
        self.assertEqual(ball, Point(689, 134))

    def test_player_marker_survives_pulse_size(self):
        for width, height in [(18, 11), (44, 24), (50, 28)]:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.fillConvexPoly(frame, np.array([[700-width//2, 500],
                [700+width//2, 500], [700, 500+height]]), (90, 180, 250))
            marker = player_marker(frame)
            self.assertIsNotNone(marker)
            self.assertAlmostEqual(marker.x, 700, delta=1)

    def test_finds_colour_blob(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[20:30, 40:50] = [240, 180, 30]
        point = colour_centroid(frame, [240, 180, 30], tolerance=5, min_pixels=20)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.x, 44.5)
        self.assertAlmostEqual(point.y, 24.5)

    def test_rejects_small_blob(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[2:4, 2:4] = [255, 255, 255]
        self.assertIsNone(colour_centroid(frame, [255, 255, 255], 1, min_pixels=5))

    def test_motion_detector_ignores_static_target(self):
        old = np.zeros((300, 400, 3), dtype=np.uint8)
        new = old.copy()
        old[100:120, 100:120] = [255, 150, 20]
        new[100:120, 100:120] = [255, 150, 20]  # static orange target
        old[180:195, 180:195] = [245, 245, 245]
        new[160:175, 190:205] = [245, 245, 245]  # moving white ball
        ball = detect_moving_ball(new, old, None, Point(190, 220))
        self.assertIsNotNone(ball)
        self.assertAlmostEqual(ball.x, 197, delta=5)
        self.assertAlmostEqual(ball.y, 167, delta=5)

    def test_tracker_blacklists_stationary_circle(self):
        tracker = Tracker()
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        vision = {"detector": "screen", "ball_roi": [0, 1, 0, 1]}
        with patch("spike_bot.vision.player_marker", return_value=Point(200, 250)), patch(
            "spike_bot.vision.detect_ball", return_value=Point(50, 60)
        ):
            observations = [tracker.observe(frame, vision, timestamp=index * 0.05) for index in range(5)]
        self.assertIsNone(observations[-1].ball)
        self.assertEqual(tracker._excluded_circles[-1], Point(50, 60))

    def test_fixed_life_indicator_is_excluded_immediately(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        with patch("spike_bot.vision._circle_candidates", return_value=[(44, 60, 20)]):
            ball = detect_ball(
                frame,
                None,
                Point(200, 250),
                [0, 1, 0, 1],
                fixed_excluded=[[0.11, 0.20]],
            )
        self.assertIsNone(ball)


if __name__ == "__main__":
    unittest.main()
