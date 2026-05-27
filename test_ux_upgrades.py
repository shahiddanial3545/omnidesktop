import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock UI libs and pyttsx3
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()
sys.modules['pyttsx3'] = MagicMock()

from habit_engine import HabitEngine

class TestUXUpgrades(unittest.TestCase):
    def setUp(self):
        self.config = {
            "habits": {
                "posture_guardian": {"enabled": True, "slouch_timeout": 0},
                "privacy_shield": {"enabled": True},
                "shush_trigger": {"enabled": True, "dist_threshold": 0.1},
                "air_scroll": {"enabled": True},
                "double_tap": {"enabled": True, "velocity_threshold": 0.01},
                "morning_routine": {"enabled": False},
                "phone_down": {"enabled": True, "roi": [0,0,1,1]},
                "coffee_mug_mute": {"enabled": True, "target_color_hsv": [0,0,0], "tolerance": 10}
            }
        }
        self.voice_mock = MagicMock()
        self.engine = HabitEngine(self.config, voice_callback=self.voice_mock)

    def test_mode_switching(self):
        self.engine.set_mode("Focus")
        self.assertEqual(self.engine.mode, "Focus")
        self.voice_mock.assert_called_with("Focus mode activated")

    def test_mode_restriction_focus(self):
        self.engine.set_mode("Focus")

        # In Focus mode, air_scroll (Lazy) should NOT trigger voice
        mock_hands = MagicMock()
        mock_hands.multi_hand_landmarks = [MagicMock()]
        mock_hands.multi_hand_landmarks[0].landmark = {0: MagicMock(y=0.1)} # Trigger condition for scroll

        self.voice_mock.reset_mock()
        self.engine.air_scroll(mock_hands)
        self.voice_mock.assert_not_called()

    def test_mode_restriction_lazy(self):
        self.engine.set_mode("Lazy")

        # In Lazy mode, posture (Focus) should NOT trigger voice
        mock_pose = MagicMock()
        mock_pose.pose_landmarks.landmark = {11: MagicMock(y=0.1), 12: MagicMock(y=0.5)} # Slouch condition

        self.voice_mock.reset_mock()
        self.engine.posture_guardian(mock_pose)
        self.voice_mock.assert_not_called()

if __name__ == '__main__':
    unittest.main()
