import json
import unittest

import numpy as np

from spike_bot.outcomes import LifeCounter, visible_lives


class OutcomeTests(unittest.TestCase):
    def test_life_count_and_json_logging(self):
        for count in range(1, 6):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            for x in (363, 414, 466, 518, 569)[:count]:
                frame[124:145, x-10:x+11] = (240, 240, 240)
                frame[124:128, x-10:x+11] = (255, 180, 0)
            self.assertEqual(visible_lives(frame), count)
            json.dumps({"lives": visible_lives(frame)})

    def test_unknown_does_not_mean_zero_or_success(self):
        self.assertIsNone(visible_lives(np.zeros((1080, 1920, 3), dtype=np.uint8)))
        counter = LifeCounter()
        for value in (5, 5, 5, None, 4, None, 5, 5, 5):
            self.assertEqual(counter.observe(value), 0)
        self.assertEqual(counter.stable, 5)
        self.assertEqual(counter.observe(4), 0)
        self.assertEqual(counter.observe(4), 0)
        self.assertEqual(counter.observe(4), 1)
        self.assertEqual(counter.observe(4), 0)


if __name__ == "__main__":
    unittest.main()
