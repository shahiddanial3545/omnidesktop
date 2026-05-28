import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock UI libs and sound
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()
sys.modules['pyttsx3'] = MagicMock()
sys.modules['winsound'] = MagicMock()

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
                "coffee_mug_mute": {"enabled": True, "target_color_hsv": [0,0,0], "tolerance": 10},
                "palm_menu": {"enabled": False},
                "gaze_dimmer": {"enabled": False}
            }
        }
        self.engine = HabitEngine(self.config)

    def test_mode_switching_chime(self):
        with patch.object(self.engine, 'play_chime') as chime_mock:
            self.engine.set_mode("Focus")
            self.assertEqual(self.engine.mode, "Focus")
            chime_mock.assert_called_with("success")

    def test_mode_restriction_focus(self):
        self.engine.set_mode("Focus")
        mock_hands = MagicMock()
        mock_hands.multi_hand_landmarks = [MagicMock()]
        mock_hands.multi_hand_landmarks[0].landmark = {0: MagicMock(y=0.1)}
        
        with patch('pyautogui.press') as press_mock:
            self.engine.air_scroll(mock_hands)
            press_mock.assert_not_called()

if __name__ == '__main__':
    unittest.main()
