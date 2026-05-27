import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import cv2
import subprocess
import shutil

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

    def _safe_macro(self, macro_func, *args, **kwargs):
        try:
            macro_func(*args, **kwargs)
        except Exception as e:
            print(f"Macro execution failed: {e}")

    def posture_guardian(self, pose_results):
        if not self.config['habits']['posture_guardian']['enabled']: return
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            shoulder_diff = abs(landmarks[11].y - landmarks[12].y)
            eye_y = (landmarks[1].y + landmarks[4].y) / 2
            if shoulder_diff > 0.05 or eye_y > 0.5:
                if not self.is_slouching:
                    self.last_slouch_time = time.time()
                    self.is_slouching = True
                if time.time() - self.last_slouch_time > self.config['habits']['posture_guardian']['slouch_timeout']:
                    self._safe_macro(sbc.set_brightness, 20)
            else:
                if self.is_slouching:
                    self._safe_macro(sbc.set_brightness, 80)
                    self.is_slouching = False

    def privacy_shield(self, face_results):
        if not self.config['habits']['privacy_shield']['enabled']: return
        if face_results and hasattr(face_results, 'detections') and face_results.detections and len(face_results.detections) > 1:
            self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def shush_trigger(self, hand_results, face_mesh_results):
        if not self.config['habits']['shush_trigger']['enabled']: return
        if hand_results.multi_hand_landmarks and face_mesh_results.multi_face_landmarks:
            # Index finger tip: landmark 8
            # Lip center (upper): landmark 13
            finger_tip = hand_results.multi_hand_landmarks[0].landmark[8]
            lip_center = face_mesh_results.multi_face_landmarks[0].landmark[13]

            dist = np.sqrt((finger_tip.x - lip_center.x)**2 + (finger_tip.y - lip_center.y)**2)
            if dist < self.config['habits']['shush_trigger']['dist_threshold']:
                self._safe_macro(pyautogui.press, 'volumemute')
                self._safe_macro(pyautogui.hotkey, 'win', 'd')

    def air_scroll(self, hand_results):
        if not self.config['habits']['air_scroll']['enabled']: return
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            wrist_y = hand_results.multi_hand_landmarks[0].landmark[0].y
            if time.time() - self.last_scroll_time > 0.2:
                if wrist_y < 0.4:
                    self._safe_macro(pyautogui.press, 'up')
                    self.last_scroll_time = time.time()
                elif wrist_y > 0.6:
                    self._safe_macro(pyautogui.press, 'down')
                    self.last_scroll_time = time.time()

    def double_tap_detector(self, hand_results):
        if not self.config['habits']['double_tap']['enabled']: return
        if hand_results.multi_hand_landmarks:
            # Wrist: landmark 0
            curr_y = hand_results.multi_hand_landmarks[0].landmark[0].y
            curr_time = time.time()

            if self.last_y_pos is not None:
                dy = curr_y - self.last_y_pos
                dt = curr_time - self.last_y_time
                if dt > 0:
                    velocity = dy / dt
                    # Sudden downward movement followed by stop (negative velocity spike)
                    if velocity > self.config['habits']['double_tap']['velocity_threshold']:
                        if curr_time - self.last_tap_time < 0.5:
                            self._safe_macro(pyautogui.press, 'space')
                            self.last_tap_time = 0 # Reset
                        else:
                            self.last_tap_time = curr_time

            self.last_y_pos = curr_y
            self.last_y_time = curr_time

    def phone_down_detector(self, frame):
        if not self.config['habits']['phone_down']['enabled']: return
        roi = self.config['habits']['phone_down']['roi']
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        phone_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(phone_roi, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < 100:
            if not self.phone_present:
                self._safe_macro(pyautogui.press, 'volumemute')
                self.phone_present = True
        else:
            if self.phone_present:
                self._safe_macro(pyautogui.press, 'volumemute')
                self.phone_present = False

    def coffee_mug_mute(self, frame):
        if not self.config['habits']['coffee_mug_mute']['enabled']: return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target = np.array(self.config['habits']['coffee_mug_mute']['target_color_hsv'])
        tol = self.config['habits']['coffee_mug_mute']['tolerance']
        lower, upper = np.clip(target - tol, 0, 255), np.clip(target + tol, 0, 255)
        mask = cv2.inRange(hsv, lower, upper)
        if cv2.countNonZero(mask) > 500:
            if not self.mug_present:
                self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm')
                self.mug_present = True
        else:
            if self.mug_present:
                self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm')
                self.mug_present = False

    def morning_routine(self, face_results):
        if not self.config['habits']['morning_routine']['enabled'] or self.morning_routine_done: return
        if face_results and hasattr(face_results, 'detections') and face_results.detections:
            for app in self.config['habits']['morning_routine']['apps']:
                if shutil.which(app): self._safe_macro(subprocess.Popen, app, shell=True)
            self.morning_routine_done = True
