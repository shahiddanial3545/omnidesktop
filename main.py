import sys
import json
import threading
import time
import cv2
from PyQt5.QtWidgets import QApplication
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard, GhostActions
from ui_bubble import OmniBubble

class OmniDeskApp:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.vision = VisionCore(
            camera_id=self.config['system']['camera_id'],
            width=self.config['system']['frame_width'],
            height=self.config['system']['frame_height']
        )

        self.habit_engine = HabitEngine(self.config)
        self.dashboard = PaperDashboard(
            corners=self.config['paper_dashboard']['corners'],
            buttons=self.config['paper_dashboard']['buttons']
        )
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
            if frame is None:
                continue

            results = self.vision.process(frame)

            # 1. Posture Guardian
            self.habit_engine.posture_guardian(results['pose'])

            # 2. Privacy Shield
            self.habit_engine.privacy_shield(results['face_detection'])

            # 3. Air Scroll
            self.habit_engine.air_scroll(results['hands'])

            # 4. Morning Routine
            self.habit_engine.morning_routine(results['face_detection'])

            # 5. Phone Down
            self.habit_engine.phone_down_detector(frame)

            # 6. Coffee Mug Mute
            self.habit_engine.coffee_mug_mute(frame)

            # 7. Paper Dashboard Tap
            if results['hands'].multi_hand_landmarks:
                # Use index finger tip of the first hand
                idx_finger = results['hands'].multi_hand_landmarks[0].landmark[8]
                tip_coords = (idx_finger.x * self.vision.width, idx_finger.y * self.vision.height)
                macro = self.dashboard.check_tap(tip_coords)
                if macro:
                    print(f"Triggering macro: {macro}")
                    self.ui.update_status(f"Action: {macro}")

            # 8. Ghost Actions
            sig = self.ghost_actions.get_signature(frame)
            motion_macro = self.ghost_actions.match_motion(sig)
            if motion_macro:
                print(f"Ghost Action: {motion_macro}")
                self.ui.update_status(f"Ghost: {motion_macro}")

            # Sleep to maintain FPS/CPU usage
            time.sleep(1.0 / self.config['system']['fps'])

    def stop(self):
        self.running = False
        if self.vision_thread:
            self.vision_thread.join()
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
