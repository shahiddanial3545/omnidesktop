import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import cv2
import subprocess
import shutil
import os

# Use winsound for Cinematic Audio Identity on Windows, with fallback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class HabitEngine:
    def __init__(self, config):
        self.config = config
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

    def set_mode(self, mode):
        self.mode = mode
        self.play_chime("success")

    def play_chime(self, type="success"):
        if not HAS_WINSOUND: return
        if type == "success": winsound.Beep(1000, 100)
        elif type == "error": winsound.Beep(400, 250)
        elif type == "detect": winsound.Beep(800, 50)

    def _safe_macro(self, macro_func, *args, **kwargs):
        try:
            macro_func(*args, **kwargs)
        except Exception as e:
            print(f"Macro failed: {e}")

    def gaze_dimmer(self, face_results):
        if self.mode == "Off" or not self.config['habits']['gaze_dimmer']['enabled']: return
        # Iris landmarks 468-472, 473-477
        if face_results and hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks:
            self.last_eye_contact = time.time()
            if self.is_dimmed:
                self._safe_macro(sbc.set_brightness, 100)
                self.is_dimmed = False
        else:
            if not self.is_dimmed and time.time() - self.last_eye_contact > self.config['habits']['gaze_dimmer']['away_timeout']:
                self._safe_macro(sbc.set_brightness, self.config['habits']['gaze_dimmer']['dim_level'])
                self.is_dimmed = True

    def palm_menu(self, hand_results):
        if self.mode == "Off" or not self.config['habits']['palm_menu']['enabled']: return
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            landmarks = hand_results.multi_hand_landmarks[0].landmark
            thumb = landmarks[4]
            tips = {"index": landmarks[8], "middle": landmarks[12], "ring": landmarks[16]}
            threshold = self.config['habits']['palm_menu']['pinch_threshold']

            for name, tip in tips.items():
                dist = np.sqrt((thumb.x - tip.x)**2 + (thumb.y - tip.y)**2)
                if dist < threshold:
                    if self.pinch_start_times[name] == 0: self.pinch_start_times[name] = time.time()
                    elif time.time() - self.pinch_start_times[name] > 0.5: # 500ms hold
                        self.play_chime("detect")
                        if name == "index": self._safe_macro(pyautogui.hotkey, 'ctrl', 't') # New tab
                        elif name == "middle": self._safe_macro(pyautogui.press, 'volumemute')
                        elif name == "ring": self._safe_macro(pyautogui.hotkey, 'win', 'd')
                        self.pinch_start_times[name] = time.time() + 2 # Cooldown
                else:
                    self.pinch_start_times[name] = 0

    def posture_guardian(self, pose_results):
        if self.mode != "Focus" and self.mode != "All": return
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            l = pose_results.pose_landmarks.landmark
            if abs(l[11].y - l[12].y) > 0.05 or (l[1].y + l[4].y)/2 > 0.5:
                if not self.is_slouching:
                    self.last_slouch_time = time.time(); self.is_slouching = True
                    self.play_chime("error")
                if time.time() - self.last_slouch_time > self.config['habits']['posture_guardian']['slouch_timeout']:
                    self._safe_macro(sbc.set_brightness, 20)
            else:
                if self.is_slouching: self._safe_macro(sbc.set_brightness, 80); self.is_slouching = False

    def privacy_shield(self, face_results):
        if self.mode != "Focus" and self.mode != "All": return
        if face_results and hasattr(face_results, 'detections') and face_results.detections and len(face_results.detections) > 1:
            self.play_chime("detect")
            self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def shush_trigger(self, hand_results, face_mesh_results):
        if self.mode != "Focus" and self.mode != "All": return
        if hand_results.multi_hand_landmarks and face_mesh_results.multi_face_landmarks:
            f = hand_results.multi_hand_landmarks[0].landmark[8]
            l = face_mesh_results.multi_face_landmarks[0].landmark[13]
            if np.sqrt((f.x-l.x)**2 + (f.y-l.y)**2) < self.config['habits']['shush_trigger']['dist_threshold']:
                self.play_chime("detect")
                self._safe_macro(pyautogui.press, 'volumemute')
                self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def air_scroll(self, hand_results):
        if self.mode != "Lazy" and self.mode != "All": return
        if hand_results.multi_hand_landmarks:
            wrist_y = hand_results.multi_hand_landmarks[0].landmark[0].y
            if time.time() - self.last_scroll_time > 0.2:
                if wrist_y < 0.4: self._safe_macro(pyautogui.press, 'up'); self.last_scroll_time = time.time()
                elif wrist_y > 0.6: self._safe_macro(pyautogui.press, 'down'); self.last_scroll_time = time.time()

    def double_tap_detector(self, hand_results):
        if self.mode == "Off": return
        if hand_results.multi_hand_landmarks:
            y = hand_results.multi_hand_landmarks[0].landmark[0].y
            t = time.time()
            if self.last_y_pos is not None:
                v = (y - self.last_y_pos) / (t - self.last_y_time) if t > self.last_y_time else 0
                if v > self.config['habits']['double_tap']['velocity_threshold']:
                    if t - self.last_tap_time < 0.5:
                        self.play_chime("success"); self._safe_macro(pyautogui.press, 'space'); self.last_tap_time = 0
                    else: self.last_tap_time = t
            self.last_y_pos = y; self.last_y_time = t

    def phone_down_detector(self, frame):
        if self.mode != "Focus" and self.mode != "All": return
        roi = self.config['habits']['phone_down']['roi']
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        fm = cv2.Laplacian(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if fm < 100:
            if not self.phone_present: self.play_chime("detect"); self._safe_macro(pyautogui.press, 'volumemute'); self.phone_present = True
        else:
            if self.phone_present: self._safe_macro(pyautogui.press, 'volumemute'); self.phone_present = False

    def coffee_mug_mute(self, frame):
        if self.mode != "Lazy" and self.mode != "All": return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target = np.array(self.config['habits']['coffee_mug_mute']['target_color_hsv'])
        tol = self.config['habits']['coffee_mug_mute']['tolerance']
        mask = cv2.inRange(hsv, np.clip(target - tol, 0, 255), np.clip(target + tol, 0, 255))
        if cv2.countNonZero(mask) > 500:
            if not self.mug_present: self.play_chime("detect"); self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm'); self.mug_present = True
        else:
            if self.mug_present: self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm'); self.mug_present = False

    def check_custom_objects(self, frame):
        for obj in self.config.get('custom_objects', []):
            target = np.array(obj['hsv'])
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.clip(target - 20, 0, 255), np.clip(target + 20, 0, 255))
            if cv2.countNonZero(mask) > 1000:
                # Custom logic here
                pass

    def morning_routine(self, face_results):
        if self.mode == "Off" or self.morning_routine_done: return
        if face_results and hasattr(face_results, 'detections') and face_results.detections:
            self.play_chime("success")
            for app in self.config['habits']['morning_routine']['apps']:
                if shutil.which(app): self._safe_macro(subprocess.Popen, app, shell=True)
            self.morning_routine_done = True
