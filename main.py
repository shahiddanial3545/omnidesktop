import sys
import json
import threading
import time
import cv2
from PyQt5.QtWidgets import QApplication
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard, GhostActions, AutoPaperDetector
from ui_bubble import OmniBubble

class OmniDeskApp:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.vision = VisionCore(
            camera_id=self.config['system']['camera_id'],
            width=self.config['system']['frame_width'],
            height=self.config['system']['frame_height'],
            alpha=self.config['system'].get('ema_alpha', 0.3)
        )

        self.habit_engine = HabitEngine(self.config)
        self.dashboard = PaperDashboard(
            corners=self.config['paper_dashboard']['corners'],
            buttons=self.config['paper_dashboard']['buttons']
        )
        self.auto_paper = AutoPaperDetector()
        self.ghost_actions = GhostActions()

        self.running = False
        self.vision_thread = None

    def start(self, ui):
        self.running = True
        self.ui = ui
        self.vision_thread = threading.Thread(target=self.run_vision, daemon=True)
        self.vision_thread.start()

    def run_vision(self):
        while self.running:
            frame = self.vision.get_frame()
            if frame is None: continue

            # 1. Auto Paper Detection (Zero Friction Setup)
            if self.config['paper_dashboard'].get('auto_detect', True):
                corners = self.auto_paper.detect(frame)
                if corners:
                    self.dashboard.set_corners(corners)

            results = self.vision.process(frame)

            # 2. Habit Triggers
            self.habit_engine.posture_guardian(results['pose'])
            self.habit_engine.privacy_shield(results['face_detection'])
            self.habit_engine.shush_trigger(results['hands'], results['face_mesh'])
            self.habit_engine.air_scroll(results['hands'])
            self.habit_engine.double_tap_detector(results['hands'])
            self.habit_engine.morning_routine(results['face_detection'])
            self.habit_engine.phone_down_detector(frame)
            self.habit_engine.coffee_mug_mute(frame)

            # 3. Spatial Macros
            if results['hands'].multi_hand_landmarks:
                idx_finger = results['hands'].multi_hand_landmarks[0].landmark[8]
                tip_coords = (idx_finger.x * self.vision.width, idx_finger.y * self.vision.height)
                macro = self.dashboard.check_tap(tip_coords)
                if macro:
                    print(f"Triggering macro: {macro}")
                    self.ui.update_status(f"Action: {macro}")

            sig = self.ghost_actions.get_signature(frame)
            motion_macro = self.ghost_actions.match_motion(sig)
            if motion_macro:
                print(f"Ghost Action: {motion_macro}")
                self.ui.update_status(f"Ghost: {motion_macro}")

            time.sleep(1.0 / self.config['system']['fps'])

    def stop(self):
        self.running = False
        if self.vision_thread: self.vision_thread.join()
        self.vision.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = OmniBubble()
    omni = OmniDeskApp()
    omni.start(ui)
    try:
        sys.exit(app.exec_())
    finally:
        omni.stop()
