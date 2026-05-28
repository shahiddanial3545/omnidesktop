import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import cv2
import subprocess
import shutil
import os
import sys

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
        self.gaze_context = [0.5, 0.5] # X, Y
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
        self._swipe_detector = None  # injected from main
        self._last_swipe_time = 0
        self._last_snap_time = 0
        self._snap_prev_dist = None
        self._snap_stable_count = 0
        self._circle_detector = None  # injected from main
        self._alt_tab_open = False
        self._last_switcher_time = 0
        self._switcher_prev_x = None
        self._rotation_detector = None  # injected from main
        self._last_volume_time = 0
        self._volume_step = 2  # percent per rotation
        self._last_media_time = 0
        self._thumb_state_prev = None
        self._air_mouse_enabled = False
        self._mouse_smooth_x = None
        self._mouse_smooth_y = None
        self._mouse_alpha = 0.4
        self._was_pinching = False
        self._was_two_pinching = False
        self._last_click_time = 0
        pyautogui.FAILSAFE = False

        # Feature A: Blink Detection
        self._blink_count = 0
        self._blink_window_start = time.time()
        self._low_blink_consecutive_minutes = 0
        self._last_blink_reminder = 0
        self._is_blinking = False

        # Feature C: Ambient Light
        self._light_mode_time = 0
        self._last_light_mode_change = 0
        self._light_baseline = []

        # Feature D: Focus Score
        self._session_start = None
        self._session_events = []
        self._session_blinks = 0
        self._session_task_switches = 0

        # Feature B: Voice Command Mode
        self._voice_thread = None
        self._voice_enabled = False

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
        old_mode = self.mode
        self.mode = mode
        self.log_event(f"Mode changed to: {mode}")
        self.play_chime("success")

        if old_mode == "Focus" and mode != "Focus":
            self.calculate_focus_score()
        elif mode == "Focus":
            self._session_start = time.time()
            self._session_events = []

    def calculate_focus_score(self):
        if not self._session_start: return
        duration_mins = (time.time() - self._session_start) / 60
        score = 100

        posture_count = self._session_events.count("posture")
        privacy_count = self._session_events.count("privacy")
        phone_count = self._session_events.count("phone")

        # Penalties
        score -= min(posture_count * 10, 40)
        score -= privacy_count * 5
        score -= phone_count * 15

        # Blink Rate Factor (Fatigue Index)
        # Normal is 12-15 blinks/min. < 8 or > 25 is bad for focus/fatigue.
        avg_blink_rate = self._session_blinks / (duration_mins + 1e-6)
        if avg_blink_rate < 8: score -= 10
        elif avg_blink_rate > 25: score -= 15

        # Task Switching Factor
        score -= min(self._session_task_switches * 5, 30)

        score = max(0, min(100, score))

        if self.stats:
            self.stats.log_stat("focus_score", score)
        self.log_event(f"Focus session ended ({int(duration_mins)}m). Score: {score}/100. Avg Blinks: {int(avg_blink_rate)}/m")

        # Suggested Break
        if score < 60:
            self.speak("Focus score is low. You might need a cognitive break.")

        self._session_start = None

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

            # Blink Detection logic
            landmarks = face_results.multi_face_landmarks[0].landmark
            # 159: upper eyelid, 145: lower eyelid
            dist = np.sqrt((landmarks[159].x - landmarks[145].x)**2 + (landmarks[159].y - landmarks[145].y)**2)
            if dist < 0.01: # Threshold for blink
                if not self._is_blinking:
                    self._blink_count += 1
                    self._is_blinking = True
            else:
                self._is_blinking = False

            now = time.time()
            if now - self._blink_window_start > 60: # 1 minute window
                if self._blink_count < 10: # Low blink rate
                    self._low_blink_consecutive_minutes += 1
                else:
                    self._low_blink_consecutive_minutes = 0

                if self._low_blink_consecutive_minutes >= 2:
                    if now - self._last_blink_reminder > 300: # 5 minute cooldown
                        self.speak("Remember to blink")
                        self.log_event("👁 Blink reminder issued")
                        self._last_blink_reminder = now

                self._blink_count = 0
                self._blink_window_start = now
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
                    if name == "index":
                        self._safe_macro(pyautogui.hotkey, 'ctrl', 't')
                        self.log_event(f"🖐 Palm menu: {name} pinch")
                    elif name == "middle":
                        self._safe_macro(pyautogui.press, 'volumemute')
                        self.log_event(f"🖐 Palm menu: {name} pinch")
                    elif name == "ring":
                        self._safe_macro(pyautogui.hotkey, 'win', 'd')
                        self.log_event(f"🖐 Palm menu: {name} pinch")
                        self.pinch_start_times[name] = time.time() + 2
                else:
                    self.pinch_start_times[name] = 0

    def posture_guardian(self, pose_results):
        if self.mode != "Focus" and self.mode != "All": return
        if not self.config.get('habits', {}).get('posture_guardian', {}).get('enabled', True): return
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            l = pose_results.pose_landmarks.landmark

            # Predictive slouch check
            slouch_ratio = (l[1].y + l[4].y)/2
            shoulder_tilt = abs(l[11].y - l[12].y)

            # Pre-emption Alert: alert if approaching limit
            if 0.45 < slouch_ratio < 0.5 and not self.is_slouching:
                 if time.time() % 30 < 1: # Throttle pre-emption logs
                    self.log_event("⚠️ Posture warning: Head dipping low")

            if shoulder_tilt > 0.05 or slouch_ratio > 0.5:
                if not self.is_slouching:
                    self.last_slouch_time = time.time(); self.is_slouching = True
                    self.speak("Please fix your posture")
                    self.log_event("Slouching detected")
                    self._session_events.append("posture")
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
                self.log_event("🔒 Privacy Shield activated")
                self._session_events.append("privacy")
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
                    self.log_event("🤫 Shush Trigger fired")
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
                        self.play_chime("success")
                        self._safe_macro(pyautogui.press, 'space')
                        self.log_event("👆 Double tap detected")
                        self.last_tap_time = 0
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
            if not self.phone_present:
                self.play_chime("detect")
                self._safe_macro(pyautogui.press, 'volumemute')
                self.phone_present = True
                self._session_events.append("phone")
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
                            self.log_event(f"📦 Object macro triggered")
                        except Exception as e:
                            print(f"Object macro failed: {e}")

    def update_gaze_context(self, gaze_vector):
        self.gaze_context = gaze_vector

    def virtual_desktop_switcher(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('virtual_desktop', {}).get('enabled', True): return

        # Gaze check
        if abs(self.gaze_context[0] - 0.5) > 0.3: return

        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
            wrist_x = hand_results.multi_hand_landmarks[0].landmark[0].x
            if self._swipe_detector:
                res = self._swipe_detector.update(wrist_x)
                now = time.time()
                if res == "left" and now - self._last_swipe_time > 0.5:
                    self._safe_macro(pyautogui.hotkey, 'ctrl', 'win', 'left')
                    self.log_event("🖥️ Virtual Desktop: Left")
                    self._last_swipe_time = now
                elif res == "right" and now - self._last_swipe_time > 0.5:
                    self._safe_macro(pyautogui.hotkey, 'ctrl', 'win', 'right')
                    self.log_event("🖥️ Virtual Desktop: Right")
                    self._last_swipe_time = now

    def window_snap_control(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('window_snap', {}).get('enabled', True): return

        # Intent Check: must be looking at screen
        if not self.check_gaze_intent(sensitivity=0.4): return

        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
            lm0 = hand_results.multi_hand_landmarks[0].landmark[0]
            lm1 = hand_results.multi_hand_landmarks[1].landmark[0]
            dist = np.sqrt((lm0.x - lm1.x)**2 + (lm0.y - lm1.y)**2)

            if self._snap_prev_dist is None:
                self._snap_prev_dist = dist
                self._snap_stable_count = 0
                return

            delta = dist - self._snap_prev_dist
            now = time.time()

            if now - self._last_snap_time < 1.2: # Increased cooldown for stability
                self._snap_prev_dist = dist
                return

            # Stability logic for window snap to prevent rapid firing
            if abs(delta) > 0.12:
                self._snap_stable_count += 1
                if self._snap_stable_count >= 3: # Must hold the delta for 3 frames
                    if delta > 0:
                        self._safe_macro(pyautogui.hotkey, 'win', 'up')
                        self.log_event("🗖 Window Maximized (Spread intent)")
                    else:
                        self._safe_macro(pyautogui.hotkey, 'win', 'down')
                        self.log_event("🗗 Window Minimized (Pinch intent)")
                    self._last_snap_time = now
                    self._snap_stable_count = 0
            elif (lm0.x < 0.15 or lm0.x > 0.85) and abs(delta) < 0.05:
                self._snap_stable_count += 1
                if self._snap_stable_count >= 5:
                    if lm0.x < 0.15:
                        self._safe_macro(pyautogui.hotkey, 'win', 'left')
                        self.log_event("⬅️ Window Snapped Left")
                    else:
                        self._safe_macro(pyautogui.hotkey, 'win', 'right')
                        self.log_event("➡️ Window Snapped Right")
                    self._last_snap_time = now
                    self._snap_stable_count = 0
            else:
                self._snap_stable_count = 0

            self._snap_prev_dist = dist

    def check_gaze_intent(self, sensitivity=0.3):
        """Checks if the user is looking at the screen center."""
        return abs(self.gaze_context[0] - 0.5) < sensitivity and abs(self.gaze_context[1] - 0.5) < sensitivity

    def app_switcher(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('app_switcher', {}).get('enabled', True): return

        # Gaze check
        if not self.check_gaze_intent():
            if self._alt_tab_open:
                pyautogui.press('return')
                pyautogui.keyUp('alt')
                self._alt_tab_open = False
            return

        if not hand_results or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks:
            if self._alt_tab_open:
                pyautogui.press('return')
                pyautogui.keyUp('alt')
                self._alt_tab_open = False
            return

        lm = hand_results.multi_hand_landmarks[0].landmark[8] # index tip
        if self._circle_detector and self._circle_detector.update(lm.x, lm.y):
            if not self._alt_tab_open:
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                self._alt_tab_open = True
                self.log_event("🔄 App Switcher opened")
                self._switcher_prev_x = lm.x

        if self._alt_tab_open and self._switcher_prev_x is not None:
            delta_x = lm.x - self._switcher_prev_x
            if delta_x > 0.08:
                pyautogui.press('tab')
                self._switcher_prev_x = lm.x
                self.log_event("➡️ App Switch: Next")
            elif delta_x < -0.08:
                pyautogui.hotkey('shift', 'tab')
                self._switcher_prev_x = lm.x
                self.log_event("⬅️ App Switch: Previous")

    def air_mouse(self, hand_results, screen_w, screen_h):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('air_mouse', {}).get('enabled', False): return
        if not hand_results or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks:
            return

        lms = hand_results.multi_hand_landmarks[0].landmark
        index_tip = lms[8]
        thumb = lms[4]

        # Map to screen coordinates (mirroring X)
        raw_x = (1.0 - index_tip.x) * screen_w
        raw_y = index_tip.y * screen_h

        # EMA Smoothing
        if self._mouse_smooth_x is None:
            self._mouse_smooth_x = raw_x
            self._mouse_smooth_y = raw_y
        else:
            self._mouse_smooth_x = self._mouse_alpha * raw_x + (1 - self._mouse_alpha) * self._mouse_smooth_x
            self._mouse_smooth_y = self._mouse_alpha * raw_y + (1 - self._mouse_alpha) * self._mouse_smooth_y

        pyautogui.moveTo(int(self._mouse_smooth_x), int(self._mouse_smooth_y), duration=0)

        now = time.time()
        # Left Click: Index + Thumb pinch
        dist_click = np.sqrt((index_tip.x - thumb.x)**2 + (index_tip.y - thumb.y)**2)
        is_pinching = dist_click < 0.04
        if is_pinching and not self._was_pinching and now - self._last_click_time > 0.4:
            pyautogui.click()
            self.log_event("🖱️ Air Mouse: Left Click")
            self._last_click_time = now
        self._was_pinching = is_pinching

        # Right Click: Index + Middle + Thumb pinch
        middle_tip = lms[12]
        dist_right = np.sqrt((middle_tip.x - thumb.x)**2 + (middle_tip.y - thumb.y)**2)
        is_two_pinching = dist_click < 0.04 and dist_right < 0.04
        if is_two_pinching and not self._was_two_pinching and now - self._last_click_time > 0.4:
            pyautogui.click(button='right')
            self.log_event("🖱️ Air Mouse: Right Click")
            self._last_click_time = now
        self._was_two_pinching = is_two_pinching

    def volume_dial(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('volume_dial', {}).get('enabled', True): return

        # Gaze check
        if not self.check_gaze_intent(sensitivity=0.4): return

        if not hand_results or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks:
            return

        lms = hand_results.multi_hand_landmarks[0].landmark
        if self._rotation_detector:
            direction = self._rotation_detector.update(lms)
            now = time.time()
            if now - self._last_volume_time < 0.3: return

            if direction == "clockwise":
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                self.log_event("🔊 Volume Up (dial)")
                self._last_volume_time = now
            elif direction == "anticlockwise":
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                self.log_event("🔉 Volume Down (dial)")
                self._last_volume_time = now

    def _get_thumb_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        wrist = landmarks[0]

        if thumb_tip.y < wrist.y - 0.15 and thumb_tip.y < index_mcp.y:
            return "up"
        if thumb_tip.x < wrist.x - 0.12 and abs(thumb_tip.y - wrist.y) < 0.08:
            return "left"
        if thumb_tip.x > wrist.x + 0.12 and abs(thumb_tip.y - wrist.y) < 0.08:
            return "right"
        return None

    def media_control(self, hand_results):
        if self.mode == "Off": return
        if not self.config.get('habits', {}).get('media_control', {}).get('enabled', True): return

        if not hand_results or not hasattr(hand_results, 'multi_hand_landmarks') or not hand_results.multi_hand_landmarks:
            self._thumb_state_prev = None
            return

        lms = hand_results.multi_hand_landmarks[0].landmark
        gesture = self._get_thumb_gesture(lms)
        now = time.time()

        if gesture == self._thumb_state_prev: return
        if gesture is None:
            self._thumb_state_prev = None
            return

        if now - self._last_media_time < 1.5: return

        if gesture == "up":
            pyautogui.press('playpause')
            self.log_event("⏯️ Media: Play/Pause")
        elif gesture == "left":
            pyautogui.press('prevtrack')
            self.log_event("⏮️ Media: Previous")
        elif gesture == "right":
            pyautogui.press('nexttrack')
            self.log_event("⏭️ Media: Next")

        self._thumb_state_prev = gesture
        self._last_media_time = now

    def voice_command_listener(self, callback):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            mic = sr.Microphone()
            while self._voice_enabled:
                with mic as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                try:
                    text = r.recognize_google(audio).lower()
                    if "omni" in text:
                        if "focus mode" in text: callback("Focus")
                        elif "lazy mode" in text: callback("Lazy")
                        elif "all on" in text: callback("All")
                        elif "mute" in text: self._safe_macro(pyautogui.press, 'volumemute')
                        elif "screenshot" in text: self._safe_macro(pyautogui.hotkey, 'win', 'prtscr')
                        self.log_event(f"🎙 Voice command: {text}")
                except Exception:
                    pass
        except Exception as e:
            print(f"Voice recognition error: {e}")

    def ambient_light_monitor(self, frame):
        now = time.time()
        avg_brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]

        # Adaptive Baseline
        self._light_baseline.append(avg_brightness)
        if len(self._light_baseline) > 100: self._light_baseline.pop(0)
        baseline = np.mean(self._light_baseline) if self._light_baseline else 128

        if now - self._last_light_mode_change < 60: return

        # Thresholds relative to adaptive baseline
        dark_thresh = max(30, baseline * 0.4)
        light_thresh = min(220, baseline * 1.6)

        if avg_brightness < dark_thresh:
            if self._light_mode_time == 0: self._light_mode_time = now
            elif now - self._light_mode_time > 10:
                if sys.platform == "win32":
                    subprocess.Popen("reg add HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize /v AppsUseLightTheme /t REG_DWORD /d 0 /f", shell=True)
                self.log_event("🌙 Dark mode activated (Adaptive)")
                self._last_light_mode_change = now; self._light_mode_time = 0
        elif avg_brightness > light_thresh:
            if self._light_mode_time == 0: self._light_mode_time = now
            elif now - self._light_mode_time > 10:
                if sys.platform == "win32":
                    subprocess.Popen("reg add HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize /v AppsUseLightTheme /t REG_DWORD /d 1 /f", shell=True)
                self.log_event("☀️ Light mode restored (Adaptive)")
                self._last_light_mode_change = now; self._light_mode_time = 0
        else:
            self._light_mode_time = 0

    def morning_routine(self, face_results):
        if self.mode == "Off" or self.morning_routine_done: return
        routine_cfg = self.config.get('habits', {}).get('morning_routine', {})
        if not routine_cfg.get('enabled', True): return
        if face_results and hasattr(face_results, 'detections') and face_results.detections:
            self.play_chime("success")
            for app in routine_cfg.get('apps', []):
                if shutil.which(app): self._safe_macro(subprocess.Popen, app, shell=True)
            self.morning_routine_done = True
