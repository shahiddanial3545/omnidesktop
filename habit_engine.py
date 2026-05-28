import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import cv2
import subprocess
import shutil
import os
from collections import deque

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

        # Feature A: Blink Detection
        self._blink_count = 0
        self._blink_window_start = time.time()
        self._low_blink_consecutive_minutes = 0
        self._last_blink_reminder = 0
        self._is_blinking = False

        # Feature C: Ambient Light
        self._light_mode_time = 0
        self._last_light_mode_change = 0

        # Feature D: Focus Score (EWMA — delegates to stats.ewma)
        self._session_start = None
        self._session_events = []

        # Feature B: Voice Command Mode
        self._voice_thread = None
        self._voice_enabled = False

        # Cooldown and tracking variables
        self._last_shush_time = 0
        self._last_privacy_time = 0
        self._obj_last_triggered = {}

        # ============================================================
        # PILLAR 1 — GestureIntentEngine injected from spatial_features
        # ============================================================
        # Lazily imported to avoid circular deps; set by OmniDeskApp
        self._gesture_intent = None   # GestureIntentEngine instance

        # ============================================================
        # PILLAR 3 — Gaze state (updated by main loop via set_gaze())
        # ============================================================
        self._gaze_on_screen = True   # assume on-screen until told otherwise
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._blink_dist = 1.0

        # ============================================================
        # PILLAR 4 — Posture pre-emption: progressive velocity check
        # ============================================================
        # We track nose-Y and shoulder symmetry over time.
        # Alert fires when velocity of slouch *approaches* threshold,
        # not just after it crosses — giving the user an early warning.
        self._pose_history = deque(maxlen=30)  # last ~1 sec at 30fps
        self._preempt_alerted = False
        self._last_preempt_time = 0

        self._tts_engine = None
        if HAS_TTS:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 160)
            except Exception:
                self._tts_engine = None

    # ------------------------------------------------------------------
    # PILLAR 3 helper — called by main loop after each face_mesh pass
    # ------------------------------------------------------------------
    def set_gaze(self, gaze_x, gaze_y, is_on_screen, blink_dist):
        self._gaze_x = gaze_x
        self._gaze_y = gaze_y
        self._gaze_on_screen = is_on_screen
        self._blink_dist = blink_dist
        # Feed eye-fatigue signal to EWMA when blink dist is very small
        # (nearly closed eyes => fatigue) — only in Focus mode
        if self.mode in ("Focus", "All") and self.stats:
            if blink_dist < 0.008:
                self.stats.record_focus_event('eye_fatigue')

    # ------------------------------------------------------------------
    # Logging & mode
    # ------------------------------------------------------------------
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
            if self.stats:
                self.stats.ewma.reset_session()

    def calculate_focus_score(self):
        """PILLAR 4 — finalise EWMA session score."""
        if not self._session_start:
            return
        if self.stats:
            score = self.stats.finalise_session()
            self.log_event(f"Focus session ended. EWMA Score: {score}/100")
        self._session_start = None

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def speak(self, text):
        self.play_chime(message=text)

    def play_chime(self, chime_type="success", message=None):
        if message and self._tts_engine:
            try:
                self._tts_engine.say(message)
                self._tts_engine.runAndWait()
                return
            except Exception:
                pass
        if HAS_WINSOUND:
            if chime_type == "success":   winsound.Beep(1000, 100)
            elif chime_type == "error":   winsound.Beep(400, 250)
            elif chime_type == "detect":  winsound.Beep(800, 50)
            elif chime_type == "warning": winsound.Beep(600, 150)

    def _safe_macro(self, macro_func, *args, **kwargs):
        try:
            macro_func(*args, **kwargs)
        except Exception as e:
            print(f"Macro failed: {e}")

    # ------------------------------------------------------------------
    # PILLAR 1 — Gesture Intent gate (used in air_scroll, shush, etc.)
    # ------------------------------------------------------------------
    def _is_intent_stable(self, hand_results):
        """
        Returns True only when GestureIntentEngine reports the hand
        velocity has been stable for ≥ intent_frames frames.
        Falls back to True if the engine is not initialised.
        """
        if self._gesture_intent is None:
            return True
        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks')
                and hand_results.multi_hand_landmarks):
            landmarks = hand_results.multi_hand_landmarks[0].landmark
            is_intentional, _ = self._gesture_intent.update(landmarks)
            return is_intentional
        return False

    # ------------------------------------------------------------------
    # Habits
    # ------------------------------------------------------------------
    def gaze_dimmer(self, face_results):
        if self.mode == "Off":
            return
        if not self.config.get('habits', {}).get('gaze_dimmer', {}).get('enabled', True):
            return
        if (face_results and hasattr(face_results, 'multi_face_landmarks')
                and face_results.multi_face_landmarks):
            self.last_eye_contact = time.time()
            if self.is_dimmed:
                self._safe_macro(sbc.set_brightness, 100)
                self.is_dimmed = False
                self.log_event("Screen brightness restored (Gaze detected)")

            # Blink Detection logic
            landmarks = face_results.multi_face_landmarks[0].landmark
            dist = np.sqrt((landmarks[159].x - landmarks[145].x)**2 +
                           (landmarks[159].y - landmarks[145].y)**2)
            if dist < 0.01:
                if not self._is_blinking:
                    self._blink_count += 1
                    self._is_blinking = True
            else:
                self._is_blinking = False

            now = time.time()
            if now - self._blink_window_start > 60:
                if self._blink_count < 10:
                    self._low_blink_consecutive_minutes += 1
                else:
                    self._low_blink_consecutive_minutes = 0

                if self._low_blink_consecutive_minutes >= 2:
                    if now - self._last_blink_reminder > 300:
                        self.speak("Remember to blink")
                        self.log_event("👁 Blink reminder issued")
                        self._last_blink_reminder = now
                        if self.stats:
                            self.stats.record_focus_event('eye_fatigue')

                self._blink_count = 0
                self._blink_window_start = now
        else:
            away_timeout = self.config.get('habits', {}).get('gaze_dimmer', {}).get('away_timeout', 5)
            if not self.is_dimmed and time.time() - self.last_eye_contact > away_timeout:
                dim_level = self.config.get('habits', {}).get('gaze_dimmer', {}).get('dim_level', 10)
                self._safe_macro(sbc.set_brightness, dim_level)
                self.is_dimmed = True
                self.log_event("Screen dimmed (No gaze detected)")

    def palm_menu(self, hand_results):
        if self.mode == "Off":
            return
        if not self.config.get('habits', {}).get('palm_menu', {}).get('enabled', True):
            return

        # PILLAR 3 — only fire if user is looking at screen
        if not self._gaze_on_screen:
            return

        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks')
                and hand_results.multi_hand_landmarks):
            landmarks = hand_results.multi_hand_landmarks[0].landmark
            thumb = landmarks[4]
            tips = {"index": landmarks[8], "middle": landmarks[12], "ring": landmarks[16]}
            threshold = self.config.get('habits', {}).get('palm_menu', {}).get('pinch_threshold', 0.05)

            for name, tip in tips.items():
                dist = np.sqrt((thumb.x - tip.x)**2 + (thumb.y - tip.y)**2)
                if dist < threshold:
                    if self.pinch_start_times[name] == 0:
                        self.pinch_start_times[name] = time.time()
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
        """
        PILLAR 4 — Enhanced with pre-emption alert.

        Normal alert fires when posture is already bad.
        Pre-emption alert fires when the slouch *velocity* is high
        (user is transitioning quickly into a bad posture) — giving
        them a heads-up before they fully slouch.
        """
        if self.mode not in ("Focus", "All"):
            return
        if not self.config.get('habits', {}).get('posture_guardian', {}).get('enabled', True):
            return
        if not (pose_results and hasattr(pose_results, 'pose_landmarks')
                and pose_results.pose_landmarks):
            self._pose_history.clear()
            return

        l = pose_results.pose_landmarks.landmark
        shoulder_asym = abs(l[11].y - l[12].y)
        nose_y = l[0].y

        # Build pose feature vector: [shoulder_asym, nose_y]
        self._pose_history.append((time.time(), shoulder_asym, nose_y))

        # ---- Pre-emption: velocity check ----
        if len(self._pose_history) >= 10:
            recent = list(self._pose_history)[-10:]
            t0, sa0, ny0 = recent[0]
            t1, sa1, ny1 = recent[-1]
            dt = max(t1 - t0, 0.001)
            # velocity of nose moving downward (increasing y) + shoulder asymmetry growing
            v_nose = (ny1 - ny0) / dt
            v_asym = (sa1 - sa0) / dt

            preempt_vel_threshold = 0.05  # normalised units/sec
            now = time.time()
            if (v_nose > preempt_vel_threshold or v_asym > preempt_vel_threshold):
                if not self._preempt_alerted and now - self._last_preempt_time > 10.0:
                    self._preempt_alerted = True
                    self._last_preempt_time = now
                    self.play_chime("warning")
                    self.log_event("⚠️ Posture pre-emption: slouch velocity rising")
                    if self.stats:
                        self.stats.record_focus_event('posture_break')
            else:
                self._preempt_alerted = False

        # ---- Standard slouch detection ----
        is_bad_posture = (shoulder_asym > 0.05 or nose_y > 0.5)
        if is_bad_posture:
            if not self.is_slouching:
                self.last_slouch_time = time.time()
                self.is_slouching = True
                self.speak("Please fix your posture")
                self.log_event("Slouching detected")
                self._session_events.append("posture")
                if self.stats:
                    self.stats.log_stat("posture_alerts", 1)
                    self.stats.record_focus_event('posture_break')
            slouch_timeout = self.config.get('habits', {}).get('posture_guardian', {}).get('slouch_timeout', 600)
            if time.time() - self.last_slouch_time > slouch_timeout:
                self._safe_macro(sbc.set_brightness, 20)
        else:
            if self.is_slouching:
                self._safe_macro(sbc.set_brightness, 100)
                self.is_slouching = False
                self.log_event("Posture corrected")

    def privacy_shield(self, face_results):
        if self.mode not in ("Focus", "All"):
            return
        if not self.config.get('habits', {}).get('privacy_shield', {}).get('enabled', True):
            return
        if (face_results and hasattr(face_results, 'detections')
                and face_results.detections and len(face_results.detections) > 1):
            now = time.time()
            if now - self._last_privacy_time > 5.0:
                self._last_privacy_time = now
                self.log_event("🔒 Privacy Shield activated")
                self._session_events.append("privacy")
                self.speak("Privacy shield activated")
                self._safe_macro(pyautogui.hotkey, 'win', 'd')
                if self.stats:
                    self.stats.record_focus_event('privacy_trigger')

    def shush_trigger(self, hand_results, face_mesh_results):
        if self.mode not in ("Focus", "All"):
            return
        if not self.config.get('habits', {}).get('shush_trigger', {}).get('enabled', True):
            return

        # PILLAR 1 — require intentional stable gesture
        if not self._is_intent_stable(hand_results):
            return

        # PILLAR 3 — only fire if user is looking at screen
        if not self._gaze_on_screen:
            return

        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks')
                and hand_results.multi_hand_landmarks
                and face_mesh_results and hasattr(face_mesh_results, 'multi_face_landmarks')
                and face_mesh_results.multi_face_landmarks):
            f = hand_results.multi_hand_landmarks[0].landmark[8]
            l = face_mesh_results.multi_face_landmarks[0].landmark[13]
            dist = np.sqrt((f.x - l.x)**2 + (f.y - l.y)**2)
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
        if self.mode not in ("Lazy", "All"):
            return
        if not self.config.get('habits', {}).get('air_scroll', {}).get('enabled', True):
            return

        # PILLAR 3 — Air Scroll only when gaze is directed at screen
        if not self._gaze_on_screen:
            return

        # PILLAR 1 — require intentional stable hand position
        if not self._is_intent_stable(hand_results):
            return

        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks')
                and hand_results.multi_hand_landmarks):
            wrist_y = hand_results.multi_hand_landmarks[0].landmark[0].y
            if time.time() - self.last_scroll_time > 0.2:
                if wrist_y < 0.4:
                    self._safe_macro(pyautogui.press, 'up')
                    self.last_scroll_time = time.time()
                elif wrist_y > 0.6:
                    self._safe_macro(pyautogui.press, 'down')
                    self.last_scroll_time = time.time()

    def double_tap_detector(self, hand_results):
        if self.mode == "Off":
            return
        if not self.config.get('habits', {}).get('double_tap', {}).get('enabled', True):
            return
        if (hand_results and hasattr(hand_results, 'multi_hand_landmarks')
                and hand_results.multi_hand_landmarks):
            y = hand_results.multi_hand_landmarks[0].landmark[0].y
            t = time.time()
            if self.last_y_pos is not None:
                v = (y - self.last_y_pos) / (t - self.last_y_time) if t > self.last_y_time else 0
                vel_thresh = self.config.get('habits', {}).get('double_tap', {}).get('velocity_threshold', 0.02)
                if v > vel_thresh:
                    if t - self.last_tap_time < 0.5:
                        self.play_chime("success")
                        self._safe_macro(pyautogui.press, 'space')
                        self.log_event("👆 Double tap detected")
                        self.last_tap_time = 0
                    else:
                        self.last_tap_time = t
            self.last_y_pos = y
            self.last_y_time = t

    def phone_down_detector(self, frame):
        if self.mode not in ("Focus", "All"):
            return
        phone_cfg = self.config.get('habits', {}).get('phone_down', {})
        if not phone_cfg.get('enabled', True):
            return
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
                if self.stats:
                    self.stats.record_focus_event('phone_detected')
        else:
            if self.phone_present:
                self._safe_macro(pyautogui.press, 'volumemute')
                self.phone_present = False

    def coffee_mug_mute(self, frame):
        if self.mode not in ("Lazy", "All"):
            return
        mug_cfg = self.config.get('habits', {}).get('coffee_mug_mute', {})
        if not mug_cfg.get('enabled', True):
            return
        target = np.array(mug_cfg.get('target_color_hsv', [0, 0, 50]))
        tol = mug_cfg.get('tolerance', 20)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.clip(target - tol, 0, 255), np.clip(target + tol, 0, 255))
        if cv2.countNonZero(mask) > 500:
            if not self.mug_present:
                self.play_chime("detect")
                self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm')
                self.mug_present = True
        else:
            if self.mug_present:
                self._safe_macro(pyautogui.hotkey, 'ctrl', 'alt', 'm')
                self.mug_present = False

    def check_custom_objects(self, frame):
        now = time.time()
        for obj in self.config.get('custom_objects', []):
            target = np.array(obj['hsv'])
            mask = cv2.inRange(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                np.clip(target - 20, 0, 255),
                np.clip(target + 20, 0, 255))
            if cv2.countNonZero(mask) > 1000:
                obj_key = str(obj['hsv'])
                if now - self._obj_last_triggered.get(obj_key, 0) > 2.0:
                    self._obj_last_triggered[obj_key] = now
                    self.play_chime("detect")
                    macro = obj.get('macro', '')
                    if macro:
                        try:
                            subprocess.Popen(macro, shell=True)
                            self.log_event("📦 Object macro triggered")
                        except Exception as e:
                            print(f"Object macro failed: {e}")

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
                        if "focus mode" in text:   callback("Focus")
                        elif "lazy mode" in text:  callback("Lazy")
                        elif "all on" in text:     callback("All")
                        elif "mute" in text:       self._safe_macro(pyautogui.press, 'volumemute')
                        elif "screenshot" in text: self._safe_macro(pyautogui.hotkey, 'win', 'prtscr')
                        self.log_event(f"🎙 Voice command: {text}")
                        if self.stats and self.mode == "Focus":
                            self.stats.record_focus_event('task_switch')
                except Exception:
                    pass
        except Exception as e:
            print(f"Voice recognition error: {e}")

    def ambient_light_monitor(self, frame):
        now = time.time()
        if now - self._last_light_mode_change < 60:
            return
        avg_brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
        if avg_brightness < 50:
            if self._light_mode_time == 0:
                self._light_mode_time = now
            elif now - self._light_mode_time > 10:
                subprocess.Popen(
                    "reg add HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize "
                    "/v AppsUseLightTheme /t REG_DWORD /d 0 /f", shell=True)
                self.log_event("🌙 Dark mode activated")
                self._last_light_mode_change = now
                self._light_mode_time = 0
        elif avg_brightness > 150:
            if self._light_mode_time == 0:
                self._light_mode_time = now
            elif now - self._light_mode_time > 10:
                subprocess.Popen(
                    "reg add HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize "
                    "/v AppsUseLightTheme /t REG_DWORD /d 1 /f", shell=True)
                self.log_event("☀️ Light mode restored")
                self._last_light_mode_change = now
                self._light_mode_time = 0
        else:
            self._light_mode_time = 0

    def morning_routine(self, face_results):
        if self.mode == "Off" or self.morning_routine_done:
            return
        routine_cfg = self.config.get('habits', {}).get('morning_routine', {})
        if not routine_cfg.get('enabled', True):
            return
        if (face_results and hasattr(face_results, 'detections')
                and face_results.detections):
            self.play_chime("success")
            for app in routine_cfg.get('apps', []):
                if shutil.which(app):
                    self._safe_macro(subprocess.Popen, app, shell=True)
            self.morning_routine_done = True
