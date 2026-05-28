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
import queue
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from vision_core import VisionCore
from habit_engine import HabitEngine
from spatial_features import PaperDashboard, SkeletalTopology, ObjectLearner, AutoPaperDetector
from stats_tracker import StatsTracker
from ui_bubble import OmniBubble

class OmniDeskApp:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.load_config()
        self._validate_config()
        self.stats = StatsTracker()
        self.vision = VisionCore(camera_id=self.config['system'].get('camera_id', 0), alpha=self.config['system'].get('ema_alpha', 0.3))
        self.habit_engine = HabitEngine(self.config)
        self.habit_engine.stats = self.stats
        self.dashboard = PaperDashboard(corners=self.config['paper_dashboard'].get('corners'), buttons=self.config['paper_dashboard'].get('buttons', []))
        self.topology = SkeletalTopology(tolerance=self.config['system'].get('gesture_tolerance', 0.85))
        self.obj_learner = ObjectLearner()
        self.auto_paper = AutoPaperDetector()
        self._input_queue = queue.Queue(maxsize=1)

        self.running = False
        self.mode = "Lazy"
        self.recording_gesture = False
        self.learning_object = False
        self.db_connected = False

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f: self.config = json.load(f)
        except:
            self.config = self._get_default_config()

    def _get_default_config(self):
        return {
            "system": {"camera_id": 0, "fps": 30, "ema_alpha": 0.3, "gesture_tolerance": 0.85, "frame_skip": 0},
            "paper_dashboard": {"corners": None, "buttons": [], "auto_detect": True},
            "custom_gestures": [],
            "custom_objects": [],
            "habits": {
                "posture_guardian": {"enabled": True, "slouch_timeout": 600, "eye_dist_threshold": 0.2},
                "privacy_shield": {"enabled": True, "sensitivity": 0.5},
                "shush_trigger": {"enabled": True, "dist_threshold": 0.05},
                "air_scroll": {"enabled": True, "sensitivity": 0.1},
                "double_tap": {"enabled": True, "velocity_threshold": 0.02},
                "palm_menu": {"enabled": True, "pinch_threshold": 0.05},
                "gaze_dimmer": {"enabled": True, "away_timeout": 5, "dim_level": 10},
                "phone_down": {"enabled": True, "roi": [0.3, 0.3, 0.7, 0.7]},
                "coffee_mug_mute": {"enabled": True, "target_color_hsv": [0, 0, 50], "tolerance": 20},
                "morning_routine": {"enabled": True, "apps": ["chrome", "code"]}
            }
        }

    def _validate_config(self):
        # Ensure self.config is a dict
        if not isinstance(self.config, dict):
            self.config = self._get_default_config()
            return

        defaults = self._get_default_config()
        for key, val in defaults.items():
            if key not in self.config or not isinstance(self.config[key], type(val)):
                self.config[key] = val
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if subkey not in self.config[key] or not isinstance(self.config[key][subkey], type(subval)):
                        self.config[key][subkey] = subval

    def save_config(self):
        with open(self.config_path, 'w') as f: json.dump(self.config, f, indent=4)

    def start(self, ui):
        self.running = True; self.ui = ui
        self.update_voice_state()
        self.ui.mode_changed.connect(self.handle_mode_change)
        self.ui.record_gesture.connect(self.start_gesture_recording)
        self.ui.learn_object.connect(self.start_object_learning)
        self.ui.show_log_requested.connect(lambda: self.ui.show_log(self.habit_engine.activity_log, self.stats.get_today_stats().get("focus_score")))
        self.ui.show_stats_requested.connect(self.handle_show_stats)
        self.ui.edit_dashboard_requested.connect(self.ui.open_dashboard_editor)
        self.ui.camera_retry_requested.connect(self.handle_camera_retry)
        self.ui.pomodoro_finished.connect(lambda: self.habit_engine.speak("Pomodoro cycle complete. Take a break."))
        self.ui.config_updated.connect(self.handle_config_update)
        self.ui.request_input_signal.connect(self._handle_input_request)
        self.vision_thread = threading.Thread(target=self.run_vision, daemon=True)
        self.vision_thread.start()

    def _handle_input_request(self, title, label):
        result = self.ui.get_macro_input(title, label)
        self._input_queue.put(result)

    def _get_input_threadsafe(self, title, label):
        self.ui.request_input_signal.emit(title, label)
        try: return self._input_queue.get(timeout=30)
        except queue.Empty: return None

    def handle_mode_change(self, mode):
        self.mode = mode; self.habit_engine.set_mode(mode)
        self.ui.update_status_signal.emit(f"Mode: {mode}")

    def trigger_gesture_recording(self):
        def recording_worker():
            try:
                self.ui.update_status_signal.emit("Recording in 3s... Hold steady!")
                time.sleep(3)

                frame = self.vision.get_frame()
                if frame is None:
                    self.ui.show_message_signal.emit("Error", "Camera frame not accessible!", "warning")
                    return

                # We need to process the frame to get landmarks
                results = self.vision.process(frame, features=['hands'])
                hands = results.get('hands')

                if hands and hands.multi_hand_landmarks:
                    raw_landmarks = hands.multi_hand_landmarks[0]
                    spatial_signature = self.topology.get_signature(raw_landmarks)

                    if spatial_signature is None:
                        self.ui.show_message_signal.emit("Retry", "Hand was too blurry. Try again!", "warning")
                        return

                    user_action = self._get_input_threadsafe("AI Gesture Integration",
                        "What feature or action do you want to integrate with this gesture?\n\nExamples:\n- https://youtube.com\n- notepad.exe")

                    if user_action and user_action.strip():
                        final_macro = user_action.strip()
                        if final_macro.startswith("http://") or final_macro.startswith("https://"):
                            final_macro = f"start chrome {final_macro}"
                        elif final_macro.endswith(".exe") or " " not in final_macro:
                            # Basic auto-correct for common commands
                            if not (final_macro.startswith("start ") or final_macro.startswith("python ")):
                                final_macro = f"start {final_macro}"

                        new_gesture_entry = {
                            "name": f"Custom_Gesture_{int(time.time())}",
                            "signature": spatial_signature,
                            "macro": final_macro
                        }

                        if "custom_gestures" not in self.config:
                            self.config["custom_gestures"] = []

                        self.config["custom_gestures"].append(new_gesture_entry)
                        self.save_config()

                        self.ui.show_message_signal.emit("Success!", f"Gesture successfully mapped to: {final_macro}", "info")
                    else:
                        self.ui.update_status_signal.emit("Recording Cancelled")
                else:
                    self.ui.show_message_signal.emit("No Hand Detected", "Webcam could not find your hand.", "warning")
            except Exception as e:
                print(f"Error compiling custom gesture profile: {e}")
                self.ui.show_message_signal.emit("Error", f"Recording failed: {str(e)}", "error")

        threading.Thread(target=recording_worker, daemon=True).start()

    def handle_show_stats(self):
        self.stats.generate_report()
        self.habit_engine.speak("Stats report generated")
        import webbrowser
        webbrowser.open('file://' + os.path.realpath('stats.html'))

    def handle_camera_retry(self):
        if self.vision.restart_camera(self.config['system'].get('camera_id', 0)):
            self.ui.update_status_signal.emit("Camera Reconnected")
            self.habit_engine.speak("Camera reconnected")
        else:
            self.ui.request_camera_error_signal.emit()

    def start_gesture_recording(self):
        self.trigger_gesture_recording()

    def handle_config_update(self, new_config):
        self.config = new_config
        self.save_config()
        self.habit_engine.config = new_config
        self.dashboard.buttons = self.config['paper_dashboard'].get('buttons', [])
        self.topology.tolerance = self.config['system'].get('gesture_tolerance', 0.85)
        self.update_voice_state()

    def update_voice_state(self):
        enabled = self.config.get('system', {}).get('voice_enabled', False)
        if enabled and not self.habit_engine._voice_enabled:
            self.habit_engine._voice_enabled = True
            self.habit_engine._voice_thread = threading.Thread(target=self.habit_engine.voice_command_listener, args=(self.handle_mode_change,), daemon=True)
            self.habit_engine._voice_thread.start()
            self.ui.update_status_signal.emit("Voice Mode ON")
        elif not enabled and self.habit_engine._voice_enabled:
            self.habit_engine._voice_enabled = False
            self.ui.update_status_signal.emit("Voice Mode OFF")

    def start_object_learning(self): self.learning_object = True; self.ui.update_status_signal.emit("Hold object in center...")

    def run_vision(self):
        frame_count = 0
        last_minute_tick = time.time()
        camera_error_shown = False
        while self.running:
            if time.time() - last_minute_tick > 60:
                if self.mode == "Focus" or self.mode == "All":
                    self.stats.log_stat("focus_minutes", 1)
                last_minute_tick = time.time()

            if self.mode == "Off": time.sleep(0.5); continue

            frame = self.vision.get_frame()
            if frame is None:
                if not camera_error_shown:
                    self.ui.request_camera_error_signal.emit()
                    camera_error_shown = True
                time.sleep(1.0); continue
            camera_error_shown = False

            frame_count += 1

            # 1. CORE TRACKING (Hands) - High frequency for gestures/scrolling
            results = self.vision.process(frame, features=['hands'])
            hands = results.get('hands')

            # 2. HEAVY HABITS - Throttled for CPU saving
            # Face Mesh / Gaze / Shush (Every 3 frames)
            if frame_count % 3 == 0:
                results.update(self.vision.process(frame, features=['face_mesh']))
                self.habit_engine.gaze_dimmer(results.get('face_mesh'))
                self.habit_engine.shush_trigger(hands, results.get('face_mesh'))

            # Posture / Privacy / Morning Routine (Every 10 frames)
            if frame_count % 10 == 0:
                results.update(self.vision.process(frame, features=['pose', 'face_detection']))
                self.habit_engine.posture_guardian(results.get('pose'))
                self.habit_engine.privacy_shield(results.get('face_detection'))
                self.habit_engine.morning_routine(results.get('face_detection'))

            # Dashboard Auto-Detect (Every 30 frames)
            if self.config['paper_dashboard'].get('auto_detect', True) and frame_count % 30 == 0:
                corners = self.auto_paper.detect(frame)
                if corners:
                    if not self.db_connected: self.habit_engine.speak("Dashboard connected"); self.db_connected = True
                    self.dashboard.set_corners(corners)
                else:
                    if self.db_connected: self.db_connected = False; self.dashboard.set_corners([])

            # 3. OBJECT & SENSOR DETECTION (Every 5 frames)
            if frame_count % 5 == 0:
                self.habit_engine.phone_down_detector(frame)
                self.habit_engine.coffee_mug_mute(frame)
                self.habit_engine.check_custom_objects(frame)

            # Ambient Light Monitor (Every 15 frames)
            if frame_count % 15 == 0:
                self.habit_engine.ambient_light_monitor(frame)

            # 4. GESTURE & INTERACTION (Immediate)
            # Old recording logic removed as it's now handled by trigger_gesture_recording worker thread

            if self.learning_object:
                profile = self.obj_learner.get_hsv_profile(frame)
                self.learning_object = False; self.habit_engine.play_chime("success")
                cmd = self._get_input_threadsafe("Object Learned", "Enter Command/URL:")
                if cmd: self.config['custom_objects'].append({"hsv": profile, "macro": cmd}); self.save_config(); self.ui.update_status_signal.emit("Object Saved")

            self.habit_engine.palm_menu(hands)
            self.habit_engine.air_scroll(hands)
            self.habit_engine.double_tap_detector(hands)

            if hands and hands.multi_hand_landmarks:
                curr_sig = self.topology.get_signature(hands.multi_hand_landmarks[0])
                for g in self.config.get('custom_gestures', []):
                    if self.topology.match(curr_sig, g['signature']) > self.topology.tolerance:
                        self.stats.log_stat("gestures_used", 1)
                        self.habit_engine.play_chime("detect"); subprocess.Popen(g['macro'], shell=True); time.sleep(2)

                idx = hands.multi_hand_landmarks[0].landmark[8]
                macro = self.dashboard.check_tap((idx.x * self.vision.width, idx.y * self.vision.height))
                if macro:
                    self.stats.log_stat("dashboard_taps", 1)
                    self.habit_engine.play_chime("detect"); subprocess.Popen(macro, shell=True); time.sleep(1)

            if self.ui.preview.isVisible():
                p_f = frame.copy()
                if hands and hands.multi_hand_landmarks:
                    for lm in hands.multi_hand_landmarks[0].landmark: cv2.circle(p_f, (int(lm.x*self.vision.width), int(lm.y*self.vision.height)), 3, (0,255,0), -1)
                if self.dashboard.corners:
                    for pt in self.dashboard.corners: cv2.circle(p_f, tuple(map(int, pt)), 5, (255,0,0), -1)
                self.ui.update_preview_signal.emit(cv2.cvtColor(p_f, cv2.COLOR_BGR2RGB))

            time.sleep(1.0 / self.config['system'].get('fps', 30))

    def stop(self): self.running = False; self.vision.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    omni = OmniDeskApp()
    ui = OmniBubble(config=omni.config)
    if not omni.config.get("system", {}).get("onboarding_done", False):
        from ui_bubble import OnboardingWizard
        wiz = OnboardingWizard(parent=ui)
        wiz.exec_()
        omni.config.setdefault("system", {})["onboarding_done"] = True
        omni.save_config()
    omni.start(ui)
    try: sys.exit(app.exec_())
    finally: omni.stop()
