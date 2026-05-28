import unittest
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['winsound'] = MagicMock()

from vision_core import EMAFilter
from habit_engine import HabitEngine
from spatial_features import SkeletalTopology

class TestUpgrades(unittest.TestCase):
    def test_ema_filter(self):
        ema = EMAFilter(alpha=0.5)
        self.assertEqual(ema.apply(10), 10)
        self.assertEqual(ema.apply(20), 15)

    def test_skeletal_topology_matching(self):
        topology = SkeletalTopology()
        
        # Identity match
        sig = [0.1] * 60 # Dummy signature
        score = topology.match(sig, sig)
        self.assertAlmostEqual(score, 1.0)

    def test_palm_menu_logic(self):
        config = {"habits": {"palm_menu": {"enabled": True, "pinch_threshold": 0.1}}}
        engine = HabitEngine(config)
        
        mock_hands = MagicMock()
        mock_hands.multi_hand_landmarks = [MagicMock()]
        # Thumb (4) and Index (8) very close
        p4 = MagicMock(x=0.5, y=0.5); p8 = MagicMock(x=0.51, y=0.51)
        mock_hands.multi_hand_landmarks[0].landmark = {4: p4, 8: p8, 12: p4, 16: p4}
        
        # Trigger first detection
        engine.palm_menu(mock_hands)
        self.assertGreater(engine.pinch_start_times["index"], 0)

if __name__ == '__main__':
    unittest.main()
