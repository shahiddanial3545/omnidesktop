import unittest
import numpy as np
import cv2
import json
import time
import os
import psutil
from unittest.mock import MagicMock, patch

# Mock dependencies
import sys
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['pyttsx3'] = MagicMock()
sys.modules['winsound'] = MagicMock()

# Import our modules
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard

class TestOmniDeskSystem(unittest.TestCase):
    def setUp(self):
        self.config = {
            "system": {"camera_id": 0, "frame_width": 640, "frame_height": 480, "fps": 30, "ema_alpha": 0.3},
            "paper_dashboard": {"corners": None, "buttons": [{"name": "Mute", "rect": [0,0,0.1,0.1], "macro": "mute"}], "auto_detect": True},
            "habits": {
                "posture_guardian": {"enabled": True, "slouch_timeout": 600},
                "phone_down": {"enabled": True, "roi": [0,0,1,1]},
                "palm_menu": {"enabled": True, "pinch_threshold": 0.05},
                "gaze_dimmer": {"enabled": True, "away_timeout": 5, "dim_level": 10}
            }
        }
        self.patcher = patch('cv2.VideoCapture')
        self.mock_cap = self.patcher.start()
        self.mock_instance = self.mock_cap.return_value
        self.mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        self.mock_instance.isOpened.return_value = True

    def tearDown(self):
        self.patcher.stop()

    def test_vision_core_initialization(self):
        vc = VisionCore()
        self.assertIsNotNone(vc.hands)
        self.assertIsNotNone(vc.pose)
        vc.release()

    def test_habit_engine_logic(self):
        engine = HabitEngine(self.config)
        engine.set_mode("Focus")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.phone_down_detector(dummy_frame)
        self.assertTrue(engine.phone_present)

    def test_memory_usage(self):
        process = psutil.Process(os.getpid())
        vc = VisionCore()
        engine = HabitEngine(self.config)
        engine.set_mode("Focus")
        for _ in range(5):
            frame = vc.get_frame()
            if frame is not None:
                results = vc.process(frame)
                engine.posture_guardian(results.get('pose'))
        final_mem = process.memory_info().rss / 1024 / 1024
        print(f"Memory Usage: {final_mem:.2f} MB")
        self.assertLess(final_mem, 500)

if __name__ == '__main__':
    unittest.main()
