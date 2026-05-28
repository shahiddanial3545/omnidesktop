import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
import json
import threading
import time
import cv2
import subprocess
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard, SkeletalTopology, ObjectLearner, AutoPaperDetector
from ui_bubble import OmniBubble

class OmniDeskApp:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.load_config()
        self.vision = VisionCore(camera_id=self.config['system'].get('camera_id', 0), alpha=self.config['system'].get('ema_alpha', 0.3))
        self.habit_engine = HabitEngine(self.config)
        self.dashboard = PaperDashboard(corners=self.config['paper_dashboard'].get('corners'), buttons=self.config['paper_dashboard'].get('buttons', []))
        self.topology = SkeletalTopology(tolerance=self.config['system'].get('gesture_tolerance', 0.85))
        self.obj_learner = ObjectLearner()
        self.auto_paper = AutoPaperDetector()

        self.running = False
        self.mode = "Lazy"
        self.recording_gesture = False
        self.learning_object = False
        self.db_connected = False

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f: self.config = json.load(f)
        except:
            self.config = {"system": {"camera_id": 0, "fps": 30, "ema_alpha": 0.3, "gesture_tolerance": 0.85}, "paper_dashboard": {"corners": None, "buttons": [], "auto_detect": True}, "custom_gestures": [], "custom_objects": [], "habits": {"posture_guardian": {"enabled": True, "slouch_timeout": 600}, "privacy_shield": {"enabled": True}, "shush_trigger": {"enabled": True, "dist_threshold": 0.05}, "air_scroll": {"enabled": True}, "double_tap": {"enabled": True, "velocity_threshold": 0.02}, "palm_menu": {"enabled": True, "pinch_threshold": 0.05}, "gaze_dimmer": {"enabled": True, "away_timeout": 5, "dim_level": 10}}}

    def save_config(self):
        with open(self.config_path, 'w') as f: json.dump(self.config, f, indent=4)

    def start(self, ui):
        self.running = True; self.ui = ui
        self.ui.mode_changed.connect(self.handle_mode_change)
        self.ui.record_gesture.connect(self.start_gesture_recording)
        self.ui.learn_object.connect(self.start_object_learning)
        self.vision_thread = threading.Thread(target=self.run_vision, daemon=True)
        self.vision_thread.start()

    def handle_mode_change(self, mode): self.mode = mode; self.habit_engine.set_mode(mode); self.ui.update_status(f"Mode: {mode}")
    def start_gesture_recording(self): self.recording_gesture = True; self.ui.update_status("Perform gesture in 3s...")
    def start_object_learning(self): self.learning_object = True; self.ui.update_status("Hold object in center...")

    def run_vision(self):
        while self.running:
            if self.mode == "Off": time.sleep(0.5); continue
            frame = self.vision.get_frame()
            if frame is None: time.sleep(0.1); continue

            if self.config['paper_dashboard'].get('auto_detect', True):
                corners = self.auto_paper.detect(frame)
                if corners:
                    if not self.db_connected: self.habit_engine.play_chime("success"); self.db_connected = True
                    self.dashboard.set_corners(corners)

            results = self.vision.process(frame)
            hands = results.get('hands')

            # 1. Recording Workflows
            if self.recording_gesture and hands and hands.multi_hand_landmarks:
                sig = self.topology.get_signature(hands.multi_hand_landmarks[0])
                if sig:
                    self.recording_gesture = False; self.habit_engine.play_chime("success")
                    cmd = self.ui.get_macro_input("Gesture Recorded", "Enter Command/URL:")
                    if cmd:
                        self.config['custom_gestures'].append({"signature": sig, "macro": cmd})
                        self.save_config(); self.ui.update_status("Gesture Saved")

            if self.learning_object:
                profile = self.obj_learner.get_hsv_profile(frame)
                self.learning_object = False; self.habit_engine.play_chime("success")
                cmd = self.ui.get_macro_input("Object Learned", "Enter Command/URL:")
                if cmd:
                    self.config['custom_objects'].append({"hsv": profile, "macro": cmd})
                    self.save_config(); self.ui.update_status("Object Saved")

            # 2. Parallel Tracking
            self.habit_engine.gaze_dimmer(results.get('face_mesh'))
            self.habit_engine.palm_menu(hands)
            self.habit_engine.posture_guardian(results.get('pose'))
            self.habit_engine.privacy_shield(results.get('face_detection'))
            self.habit_engine.shush_trigger(hands, results.get('face_mesh'))
            self.habit_engine.air_scroll(hands)
            self.habit_engine.double_tap_detector(hands)
            self.habit_engine.morning_routine(results.get('face_detection'))
            self.habit_engine.phone_down_detector(frame)
            self.habit_engine.coffee_mug_mute(frame)

            # 3. Custom Matchers
            if hands and hands.multi_hand_landmarks:
                curr_sig = self.topology.get_signature(hands.multi_hand_landmarks[0])
                for g in self.config.get('custom_gestures', []):
                    if self.topology.match(curr_sig, g['signature']) > self.topology.tolerance:
                        self.habit_engine.play_chime("detect")
                        subprocess.Popen(g['macro'], shell=True); time.sleep(2) # Cooldown

                idx = hands.multi_hand_landmarks[0].landmark[8]
                macro = self.dashboard.check_tap((idx.x * self.vision.width, idx.y * self.vision.height))
                if macro: self.habit_engine.play_chime("detect"); subprocess.Popen(macro, shell=True); time.sleep(1)

            if self.ui.preview.isVisible():
                p_f = frame.copy()
                if hands and hands.multi_hand_landmarks:
                    for lm in hands.multi_hand_landmarks[0].landmark: cv2.circle(p_f, (int(lm.x*self.vision.width), int(lm.y*self.vision.height)), 3, (0,255,0), -1)
                if self.dashboard.corners:
                    for pt in self.dashboard.corners: cv2.circle(p_f, tuple(map(int, pt)), 5, (255,0,0), -1)
                self.ui.preview.update_frame(cv2.cvtColor(p_f, cv2.COLOR_BGR2RGB))

            time.sleep(1.0 / self.config['system'].get('fps', 30))

    def stop(self): self.running = False; self.vision.release()

if __name__ == "__main__":
    app = QApplication(sys.argv); ui = OmniBubble(); omni = OmniDeskApp(); omni.start(ui)
    try: sys.exit(app.exec_())
    finally: omni.stop()
