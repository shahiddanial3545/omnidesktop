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

    def posture_guardian(self, pose_results):
        if not self.config['habits']['posture_guardian']['enabled']:
            return

        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            # Shoulder alignment (landmarks 11, 12)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Simple check: vertical difference between shoulders
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

            # Eye-to-screen distance (eye landmarks 1-6)
            eye_y = (landmarks[1].y + landmarks[4].y) / 2

            if shoulder_diff > 0.05 or eye_y > 0.5: # Arbitrary thresholds for demo
                if not self.is_slouching:
                    self.last_slouch_time = time.time()
                    self.is_slouching = True

                if time.time() - self.last_slouch_time > self.config['habits']['posture_guardian']['slouch_timeout']:
                    try:
                        sbc.set_brightness(20)
                    except:
                        pass
            else:
                if self.is_slouching:
                    try:
                        sbc.set_brightness(80)
                    except:
                        pass
                    self.is_slouching = False

    def privacy_shield(self, face_results):
        if not self.config['habits']['privacy_shield']['enabled']:
            return

        if face_results and hasattr(face_results, 'detections') and face_results.detections and len(face_results.detections) > 1:
            # More than one face detected - someone might be peeking
            pyautogui.hotkey('win', 'd') # Minimize all windows

    def air_scroll(self, hand_results):
        if not self.config['habits']['air_scroll']['enabled']:
            return

        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Wrist y position (landmark 0)
                wrist_y = hand_landmarks.landmark[0].y

                if time.time() - self.last_scroll_time > 0.2:
                    if wrist_y < 0.4:
                        pyautogui.press('up')
                        self.last_scroll_time = time.time()
                    elif wrist_y > 0.6:
                        pyautogui.press('down')
                        self.last_scroll_time = time.time()

    def phone_down_detector(self, frame):
        if not self.config['habits']['phone_down']['enabled']:
            return

        roi = self.config['habits']['phone_down']['roi']
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        phone_roi = frame[y1:y2, x1:x2]

        # Simple laplacian variance for "face-down"
        gray = cv2.cvtColor(phone_roi, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < 100: # Threshold for "flat/uninteresting" surface in ROI
            if not self.phone_present:
                pyautogui.press('volumemute') # Placeholder for DND
                self.phone_present = True
        else:
            if self.phone_present:
                pyautogui.press('volumemute')
                self.phone_present = False

    def coffee_mug_mute(self, frame):
        if not self.config['habits']['coffee_mug_mute']['enabled']:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target = np.array(self.config['habits']['coffee_mug_mute']['target_color_hsv'])
        tol = self.config['habits']['coffee_mug_mute']['tolerance']

        lower = np.clip(target - tol, 0, 255)
        upper = np.clip(target + tol, 0, 255)

        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > 500: # Mug detected in frame
            if not self.mug_present:
                # Mute mic (platform dependent, using shortcut or generic command)
                pyautogui.hotkey('ctrl', 'alt', 'm')
                self.mug_present = True
        else:
            if self.mug_present:
                pyautogui.hotkey('ctrl', 'alt', 'm')
                self.mug_present = False

    def morning_routine(self, face_results):
        if not self.config['habits']['morning_routine']['enabled'] or self.morning_routine_done:
            return

        if face_results and hasattr(face_results, 'detections') and face_results.detections:
            # First face of the day!
            for app in self.config['habits']['morning_routine']['apps']:
                # Try to find executable in PATH
                if shutil.which(app):
                    try:
                        subprocess.Popen(app, shell=True)
                    except:
                        pass
            self.morning_routine_done = True
