import sys
import json
import threading
import time
import cv2

# Global flag for pyttsx3 availability
PYTTSX3_AVAILABLE = False
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard, GhostActions, AutoPaperDetector
from ui_bubble import OmniBubble

class VoiceManager:
    def __init__(self):
        self.engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
            except Exception as e:
                print(f"Voice engine init failed: {e}")
                self.engine = None

    def speak(self, text):
        if self.engine:
            def _speak():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            threading.Thread(target=_speak, daemon=True).start()

class OmniDeskApp:
    def __init__(self, config_path='config.json'):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            # Fallback default config if file is missing/broken
            self.config = {
                "system": {"camera_id": 0, "frame_width": 640, "frame_height": 480, "fps": 30, "ema_alpha": 0.3},
                "paper_dashboard": {"corners": None, "buttons": [], "auto_detect": True},
                "habits": {
                    "posture_guardian": {"enabled": True, "slouch_timeout": 600},
                    "privacy_shield": {"enabled": True},
                    "shush_trigger": {"enabled": True, "dist_threshold": 0.05},
                    "air_scroll": {"enabled": True},
                    "double_tap": {"enabled": True, "velocity_threshold": 0.02},
                    "morning_routine": {"enabled": False},
                    "phone_down": {"enabled": False},
                    "coffee_mug_mute": {"enabled": False}
                }
            }

        self.voice = VoiceManager()
        self.vision = VisionCore(
            camera_id=self.config['system'].get('camera_id', 0),
            width=self.config['system'].get('frame_width', 640),
            height=self.config['system'].get('frame_height', 480),
            alpha=self.config['system'].get('ema_alpha', 0.3)
        )

        self.habit_engine = HabitEngine(self.config, voice_callback=self.voice.speak)
        self.dashboard = PaperDashboard(
            corners=self.config['paper_dashboard'].get('corners'),
            buttons=self.config['paper_dashboard'].get('buttons', [])
        )
        self.auto_paper = AutoPaperDetector()
        self.ghost_actions = GhostActions()

        self.running = False
        self.vision_thread = None
        self.mode = "Lazy"
        self.dashboard_connected = False

    def start(self, ui):
        self.running = True
        self.ui = ui
        self.ui.mode_changed.connect(self.handle_mode_change)
        self.vision_thread = threading.Thread(target=self.run_vision, daemon=True)
        self.vision_thread.start()

    def handle_mode_change(self, mode):
        self.mode = mode
        self.habit_engine.set_mode(mode)
        self.ui.update_status(f"Mode: {mode}")

    def run_vision(self):
        while self.running:
            if self.mode == "Off":
                time.sleep(1)
                continue

            frame = self.vision.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # 1. Auto Paper Detection
            if self.config['paper_dashboard'].get('auto_detect', True):
                corners = self.auto_paper.detect(frame)
                if corners:
                    if not self.dashboard_connected:
                        self.voice.speak("Dashboard connected")
                        self.dashboard_connected = True
                    self.dashboard.set_corners(corners)

            results = self.vision.process(frame)

            # 2. Habit Triggers (Check keys to prevent KeyError if results are dummy)
            self.habit_engine.posture_guardian(results.get('pose'))
            self.habit_engine.privacy_shield(results.get('face_detection'))
            self.habit_engine.shush_trigger(results.get('hands'), results.get('face_mesh'))
            self.habit_engine.air_scroll(results.get('hands'))
            self.habit_engine.double_tap_detector(results.get('hands'))
            self.habit_engine.morning_routine(results.get('face_detection'))
            self.habit_engine.phone_down_detector(frame)
            self.habit_engine.coffee_mug_mute(frame)

            # 3. Spatial Macros
            hands = results.get('hands')
            if hands and hasattr(hands, 'multi_hand_landmarks') and hands.multi_hand_landmarks:
                idx_finger = hands.multi_hand_landmarks[0].landmark[8]
                tip_coords = (idx_finger.x * self.vision.width, idx_finger.y * self.vision.height)
                macro = self.dashboard.check_tap(tip_coords)
                if macro:
                    self.voice.speak(f"Triggering {macro.replace('_', ' ')}")
                    self.ui.update_status(f"Action: {macro}")

            # 4. Update UI Preview
            if hasattr(self.ui, 'preview') and self.ui.preview.isVisible():
                preview_frame = frame.copy()
                if hands and hasattr(hands, 'multi_hand_landmarks') and hands.multi_hand_landmarks:
                    for lm in hands.multi_hand_landmarks[0].landmark:
                        cv2.circle(preview_frame, (int(lm.x*self.vision.width), int(lm.y*self.vision.height)), 3, (0, 255, 0), -1)
                if self.dashboard.corners:
                    for pt in self.dashboard.corners:
                        cv2.circle(preview_frame, tuple(map(int, pt)), 5, (255, 0, 0), -1)

                preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                self.ui.preview.update_frame(preview_frame)

            time.sleep(1.0 / self.config['system'].get('fps', 30))

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
