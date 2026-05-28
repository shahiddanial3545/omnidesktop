import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import cv2
import subprocess
import shutil
import os

# Robust audio fallback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

class HabitEngine:
    def __init__(self, config):
        self.config = config
        self.stats = None
        self.activity_log = []
        self.last_slouch_time = time.time()
        self.is_slouching = False
        self.morning_routine_done = False
        self.last_scroll_time = time.time()
        self.phone_present = False
        self.mug_present = False
        self.last_y_pos = None
        self.last_y_time = None
        self.last_tap_time = 0
        self.mode = "Lazy"
        self.last_eye_contact = time.time()
        self.is_dimmed = False
        self.pinch_start_times = {"index": 0, "middle": 0, "ring": 0}

        # New cooldown and tracking variables
        self._last_shush_time = 0
        self._last_privacy_time = 0
        self._obj_last_triggered = {}

        self._tts_engine = None
        if HAS_TTS:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 160)
            except Exception:
                self._tts_engine = None

    def log_event(self, event):
        timestamp = time.strftime("%H:%M:%S")
        self.activity_log.append(f"[{timestamp}] {event}")
        if len(self.activity_log) > 50:
            self.activity_log.pop(0)

    def set_mode(self, mode):
        self.mode = mode
        self.log_event(f"Mode changed to: {mode}")
        self.play_chime("success")

    def speak(self, text):
        self.play_chime(message=text)

    def play_chime(self, chime_type="success", message=None):
        """Cross-platform audio feedback. Speaks message if provided, else beeps."""
        if message and self._tts_engine:
            try:
                self._tts_engine.say(message)
                self._tts_engine.runAndWait()
                return
            except Exception:
                pass

        if HAS_WINSOUND:
            if chime_type == "success": winsound.Beep(1000, 100)
            elif chime_type == "error": winsound.Beep(400, 250)
            elif chime_type == "detect": winsound.Beep(800, 50)

    def _safe_macro(self, macro_func, *args, **kwargs):
        try:
            macro_func(*args, **kwargs)
        except Exception as e:
            print(f"Macro failed: {e}")

    def gaze_dimmer(self, face_results):
        if self.mode == "Off" or not self.config.get('habits', {}).get('gaze_dimmer', {}).get('enabled', True): return
        if face_results and hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks:
            self.last_eye_contact = time.time()
            if self.is_dimmed:
                self._safe_macro(sbc.set_brightness, 100)
                self.is_dimmed = False
                self.log_event("Screen brightness restored (Gaze detected)")
        else:
            if not self.is_dimmed and time.time() - self.last_eye_contact > self.config.get('habits', {}).get('gaze_dimmer', {}).get('away_timeout', 5):
                self._safe_macro(sbc.set_brightness, self.config.get('habits', {}).get('gaze_dimmer', {}).get('dim_level', 10))
                self.is_dimmed = True
                self.log_event("Screen dimmed (No gaze detected)")

    def palm_menu(self, hand_results):
        if self.mode == "Off" or not self.config.get('habits', {}).get('palm_menu', {}).get('enabled', True): return
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            landmarks = hand_results.multi_hand_landmarks[0].landmark
            thumb = landmarks[4]
            tips = {"index": landmarks[8], "middle": landmarks[12], "ring": landmarks[16]}
            threshold = self.config.get('habits', {}).get('palm_menu', {}).get('pinch_threshold', 0.05)

            for name, tip in tips.items():
                dist = np.sqrt((thumb.x - tip.x)**2 + (thumb.y - tip.y)**2)
                if dist < threshold:
                    if self.pinch_start_times[name] == 0: self.pinch_start_times[name] = time.time()
                    elif time.time() - self.pinch_start_times[name] > 0.5:
                        self.play_chime("detect")
                        if name == "index": self._safe_macro(pyautogui.hotkey, 'ctrl', 't')
                        elif name == "middle": self._safe_macro(pyautogui.press, 'volumemute')
                        elif name == "ring": self._safe_macro(pyautogui.hotkey, 'win', 'd')
                        self.pinch_start_times[name] = time.time() + 2
                else:
                    self.pinch_start_times[name] = 0

    def posture_guardian(self, pose_results):
        if self.mode != "Focus" and self.mode != "All": return
        if not self.config.get('habits', {}).get('posture_guardian', {}).get('enabled', True): return
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            l = pose_results.pose_landmarks.landmark
            if abs(l[11].y - l[12].y) > 0.05 or (l[1].y + l[4].y)/2 > 0.5:
                if not self.is_slouching:
                    self.last_slouch_time = time.time(); self.is_slouching = True
                    self.speak("Please fix your posture")
                    self.log_event("Slouching detected")
                    if self.stats: self.stats.log_stat("posture_alerts", 1)
                if time.time() - self.last_slouch_time > self.config.get('habits', {}).get('posture_guardian', {}).get('slouch_timeout', 600):
                    self._safe_macro(sbc.set_brightness, 20)
            else:
                if self.is_slouching:
                    self._safe_macro(sbc.set_brightness, 100)
                    self.is_slouching = False
                    self.log_event("Posture corrected")

    def privacy_shield(self, face_results):
        if self.mode != "Focus" and self.mode != "All": return
        if not self.config.get('habits', {}).get('privacy_shield', {}).get('enabled', True): return
        if face_results and hasattr(face_results, 'detections') and face_results.detections and len(face_results.detections) > 1:
            now = time.time()
            if now - self._last_privacy_time > 5.0:
                self._last_privacy_time = now
                self.speak("Privacy shield activated")
                self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def shush_trigger(self, hand_results, face_mesh_results):
        if self.mode != "Focus" and self.mode != "All": return
        if not self.config.get('habits', {}).get('shush_trigger', {}).get('enabled', True): return
        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks and
            face_mesh_results and hasattr(face_mesh_results, 'multi_face_landmarks') and face_mesh_results.multi_face_landmarks):
            f = hand_results.multi_hand_landmarks[0].landmark[8]
            l = face_mesh_results.multi_face_landmarks[0].landmark[13]
            dist = np.sqrt((f.x-l.x)**2 + (f.y-l.y)**2)
            threshold = self.config.get('habits', {}).get('shush_trigger', {}).get('dist_threshold', 0.05)
            if dist < threshold:
                now = time.time()
                if now - self._last_shush_time > 3.0:
                    self._last_shush_time = now
                    self.speak("Muting microphone")
                    self._safe_macro(pyautogui.press, 'volumemute')
                    self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def air_scroll(self, hand_results):
        if self.mode != "Lazy" and self.mode != "All": return
        if not self.config.get('habits', {}).get('air_scroll', {}).get('enabled', True): return
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            wrist_y = hand_results.multi_hand_landmarks[0].landmark[0].y
            if time.time() - self.last_scroll_time > 0.2:
                if wrist_y < 0.4: self._safe_macro(pyautogui.press, 'up'); self.last_scroll_time = time.time()
                elif wrist_y > 0.6: self._safe_macro(pyautogui.press, 'down'); self.last_scroll_time = time.time()

    def double_tap_detector(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('double_tap', {}).get('enabled', True): return
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            y = hand_results.multi_hand_landmarks[0].landmark[0].y
            t = time.time()
            if self.last_y_pos is not None:
                v = (y - self.last_y_pos) / (t - self.last_y_time) if t > self.last_y_time else 0
                if v > self.config.get('habits', {}).get('double_tap', {}).get('velocity_threshold', 0.02):
                    if t - self.last_tap_time < 0.5:
                        self.play_chime("success"); self._safe_macro(pyautogui.press, 'space'); self.last_tap_time = 0
                    else: self.last_tap_time = t
            self.last_y_pos = y; self.last_y_time = t

    def phone_down_detector(self, frame):
        if self.mode != "Focus" and self.mode != "All": return
        phone_cfg = self.config.get('habits', {}).get('phone_down', {})
        if not phone_cfg.get('enabled', True): return
        roi = phone_cfg.get('roi', [0.3, 0.3, 0.7, 0.7])
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        fm = cv2.Laplacian(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if fm < 100:
            if not self.phone_present: self.play_chime("detect"); self._safe_macro(pyautogui.press, 'volumemute'); self.phone_present = True
        else:
            if self.phone_present: self._safe_macro(pyautogui.press, 'volumemute'); self.phone_present = False

    def coffee_mug_mute(self, frame):
        if self.mode != "Lazy" and self.mode != "All": return
        mug_cfg = self.config.get('habits', {}).get('coffee_mug_mute', {})
        if not mug_cfg.get('enabled', True): return
        target = np.array(mug_cfg.get('target_color_hsv', [0, 0, 50]))
        tol = mug_cfg.get('tolerance', 20)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.clip(target - tol, 0, 255), np.clip(target + tol, 0, 255))
        if cv2.countNonZero(mask) > 500:
            if not self.mug_present: self.play_chime("detect"); self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm'); self.mug_present = True
        else:
            if self.mug_present: self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm'); self.mug_present = False

    def check_custom_objects(self, frame):
        now = time.time()
        for obj in self.config.get('custom_objects', []):
            target = np.array(obj['hsv'])
            mask = cv2.inRange(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                np.clip(target - 20, 0, 255),
                np.clip(target + 20, 0, 255)
            )
            if cv2.countNonZero(mask) > 1000:
                obj_key = str(obj['hsv'])
                last_time = self._obj_last_triggered.get(obj_key, 0)
                if now - last_time > 2.0:
                    self._obj_last_triggered[obj_key] = now
                    self.play_chime("detect")
                    macro = obj.get('macro', '')
                    if macro:
                        try:
                            import subprocess
                            subprocess.Popen(macro, shell=True)
                        except Exception as e:
                            print(f"Object macro failed: {e}")

    def morning_routine(self, face_results):
        if self.mode == "Off" or self.morning_routine_done: return
        routine_cfg = self.config.get('habits', {}).get('morning_routine', {})
        if not routine_cfg.get('enabled', True): return
        if face_results and hasattr(face_results, 'detections') and face_results.detections:
            self.play_chime("success")
            for app in routine_cfg.get('apps', []):
                if shutil.which(app): self._safe_macro(subprocess.Popen, app, shell=True)
            self.morning_routine_done = True
