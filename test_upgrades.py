import unittest
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock UI libs
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()

from vision_core import EMAFilter, VisionCore
from habit_engine import HabitEngine

class TestUpgrades(unittest.TestCase):
    def test_ema_filter(self):
        ema = EMAFilter(alpha=0.5)
        self.assertEqual(ema.apply(10), 10)
        self.assertEqual(ema.apply(20), 15)
        self.assertEqual(ema.apply(30), 22.5)

    def test_habit_robustness(self):
        config = {
            "habits": {
                "posture_guardian": {"enabled": True, "slouch_timeout": 0},
                "privacy_shield": {"enabled": True},
                "shush_trigger": {"enabled": True, "dist_threshold": 0.1},
                "air_scroll": {"enabled": True},
                "double_tap": {"enabled": True, "velocity_threshold": 0.01},
                "morning_routine": {"enabled": False},
                "phone_down": {"enabled": False},
                "coffee_mug_mute": {"enabled": False}
            }
        }
        engine = HabitEngine(config)

        # Test shush distance logic
        mock_hands = MagicMock()
        mock_hands.multi_hand_landmarks = [MagicMock()]
        mock_hands.multi_hand_landmarks[0].landmark = {8: MagicMock(x=0.5, y=0.5)}

        mock_face = MagicMock()
        mock_face.multi_face_landmarks = [MagicMock()]
        mock_face.multi_face_landmarks[0].landmark = {13: MagicMock(x=0.51, y=0.51)}

        # Should not crash
        engine.shush_trigger(mock_hands, mock_face)

    def test_double_tap_velocity(self):
        config = {"habits": {"double_tap": {"enabled": True, "velocity_threshold": 0.01}}}
        engine = HabitEngine(config)

        mock_hands = MagicMock()
        mock_hands.multi_hand_landmarks = [MagicMock()]

        # Tap 1
        mock_hands.multi_hand_landmarks[0].landmark = {0: MagicMock(y=0.5)}
        engine.double_tap_detector(mock_hands)

        import time
        time.sleep(0.1)

        # Rapid move down
        mock_hands.multi_hand_landmarks[0].landmark = {0: MagicMock(y=0.6)}
        engine.double_tap_detector(mock_hands)

        self.assertNotEqual(engine.last_tap_time, 0)

if __name__ == '__main__':
    unittest.main()
