import cv2
import numpy as np
import time
from collections import deque

# ============================================================
# PILLAR 1 — KALMAN FILTER + GESTURE INTENT ENGINE
# ============================================================

class GestureKalmanFilter:
    """
    1-D Kalman filter for a single landmark coordinate.
    Tracks position (x) and velocity (dx/dt).
    Used by GestureIntentEngine to smooth noisy MediaPipe landmarks.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        # State vector: [position, velocity]
        self.x = np.zeros(2)           # state estimate
        self.P = np.eye(2) * 1.0       # covariance estimate
        self.Q = np.eye(2) * process_noise   # process noise
        self.R = np.array([[measurement_noise]])  # measurement noise
        self.H = np.array([[1.0, 0.0]])  # observation matrix
        self.F = None                   # transition matrix (set on first update)

    def predict(self, dt=1.0):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def update(self, z, dt=1.0):
        self.predict(dt)
        y = np.array([z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0], self.x[1]   # position, velocity


class GestureIntentEngine:
    """
    PILLAR 1: Multi-landmark Kalman smoother + velocity-stability gate.

    Algorithm:
      1. Each landmark x/y/z runs through its own Kalman filter.
      2. After smoothing, compute aggregate wrist velocity magnitude.
      3. A gesture fires only if velocity < stability_threshold for
         intent_frames consecutive frames — proving deliberate intent,
         not accidental motion.
    """
    def __init__(self, n_landmarks=21, stability_threshold=0.008,
                 intent_frames=6, pn=1e-4, mn=1e-2):
        self.n = n_landmarks
        self.stability_threshold = stability_threshold
        self.intent_frames = intent_frames
        self._kf = {f"{i}_{ax}": GestureKalmanFilter(pn, mn)
                    for i in range(n_landmarks) for ax in ('x', 'y', 'z')}
        self._stable_count = 0
        self._last_t = None

    def update(self, landmarks):
        """
        Smooth landmarks in-place and return (is_intentional, velocity_mag).
        landmarks: mediapipe hand_landmarks.landmark (list-like of 21 points)
        """
        now = time.time()
        dt = (now - self._last_t) if self._last_t else 0.033
        self._last_t = now

        vels = []
        for i, lm in enumerate(landmarks):
            px, vx = self._kf[f"{i}_x"].update(lm.x, dt)
            py, vy = self._kf[f"{i}_y"].update(lm.y, dt)
            pz, vz = self._kf[f"{i}_z"].update(lm.z, dt)
            # Write smoothed values back in-place
            lm.x, lm.y, lm.z = px, py, pz
            if i == 0:  # wrist
                vels.append(np.sqrt(vx**2 + vy**2))

        vel_mag = float(np.mean(vels)) if vels else 1.0

        if vel_mag < self.stability_threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0

        is_intentional = self._stable_count >= self.intent_frames
        return is_intentional, vel_mag

    def reset(self):
        self._stable_count = 0


# ============================================================
# PILLAR 2 — ADAPTIVE AUTO-CALIBRATION
# ============================================================

class AdaptiveAutoPaperDetector:
    """
    PILLAR 2: Perspective-warp tracking with temporal smoothing.

    Enhancements over AutoPaperDetector:
      - Maintains a rolling buffer of detected corner sets.
      - Outputs the EMA-smoothed corners so minor desk shifts
        don't cause jitter.
      - Adaptive ambient-light baseline: computes rolling mean
        brightness of the paper region; suppresses false detections
        when subtle shadows pass by (avoids registry flip flicker).
    """
    def __init__(self, corner_history=10, brightness_window=30,
                 shadow_hysteresis=15.0):
        self.last_corners = None
        self._corner_buf = deque(maxlen=corner_history)
        self._brightness_buf = deque(maxlen=brightness_window)
        self._shadow_hysteresis = shadow_hysteresis   # lux units
        self._baseline_brightness = None

    # ---- private helpers ----
    @staticmethod
    def _order_points(pts):
        pts = np.array(pts, dtype="float32")
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def _smooth_corners(self):
        if not self._corner_buf:
            return None
        buf = np.array(list(self._corner_buf), dtype=float)  # (N,4,2)
        return buf.mean(axis=0).tolist()

    def _update_brightness(self, frame, corners):
        """Track mean brightness inside the paper ROI."""
        if corners is None or len(corners) != 4:
            return
        ordered = self._order_points(corners)
        dst = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(frame, M, (200, 200))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        self._brightness_buf.append(mean_b)
        if len(self._brightness_buf) == self._brightness_buf.maxlen:
            self._baseline_brightness = float(np.mean(self._brightness_buf))

    def is_shadow_event(self, frame, candidate_corners):
        """
        Returns True if the brightness delta exceeds hysteresis threshold.
        Suppresses corner updates when a passing shadow mimics paper edges.
        """
        if self._baseline_brightness is None:
            return False
        self._update_brightness(frame, candidate_corners)
        if not self._brightness_buf:
            return False
        current = self._brightness_buf[-1]
        delta = abs(current - self._baseline_brightness)
        return delta > self._shadow_hysteresis

    # ---- public API ----
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50_000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                candidate = approx.reshape(4, 2).tolist()

                # Shadow-suppression gate
                if self.is_shadow_event(frame, candidate):
                    # Use existing smoothed corners instead of jumping
                    return self._smooth_corners() or self.last_corners

                self._corner_buf.append(np.array(candidate, dtype=float))
                self._update_brightness(frame, candidate)
                self.last_corners = self._smooth_corners()
                return self.last_corners

        # No contour found — gently decay the history
        if self.last_corners is not None:
            return self._smooth_corners()
        return None


# ============================================================
# Keep original AutoPaperDetector name as alias for compatibility
# ============================================================
AutoPaperDetector = AdaptiveAutoPaperDetector


class PaperDashboard:
    def __init__(self, corners=None, buttons=None):
        self.corners = corners
        self.buttons = buttons or []
        self.matrix = None
        self._last_tap_times = {}
        if self.corners and len(self.corners) == 4:
            self._update_matrix()

    def _update_matrix(self):
        pts = np.array(self.corners, dtype="float32")
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        dst = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype="float32")
        self.matrix = cv2.getPerspectiveTransform(rect, dst)

    def set_corners(self, corners):
        self.corners = corners
        if self.corners and len(self.corners) == 4:
            self._update_matrix()
        else:
            self.matrix = None

    def check_tap(self, finger_tip, cooldown=1.5):
        if self.matrix is None or finger_tip is None:
            return None
        point = np.array([[[finger_tip[0], finger_tip[1]]]], dtype="float32")
        transformed = cv2.perspectiveTransform(point, self.matrix)[0][0]
        mx, my = transformed[0] / 1000.0, transformed[1] / 1000.0
        now = time.time()
        for btn in self.buttons:
            bx, by, bw, bh = btn['rect']
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                btn_name = btn.get('name', str(btn['rect']))
                if now - self._last_tap_times.get(btn_name, 0) > cooldown:
                    self._last_tap_times[btn_name] = now
                    return btn['macro']
        return None


class SkeletalTopology:
    """
    PILLAR 1 (enhanced): Gesture signature matching with DTW-inspired
    windowed buffer.  Still produces cosine-similarity scores, but now
    the match() method compares against a short rolling window of
    recent signatures to tolerate slight temporal misalignment.
    """
    def __init__(self, tolerance=0.85, window=5):
        self.tolerance = tolerance
        self._window = window
        self._sig_buffer = deque(maxlen=window)

    def get_signature(self, landmarks):
        if not landmarks:
            return None
        base = landmarks.landmark[0]
        points = []
        for lm in landmarks.landmark[1:]:
            points.append([lm.x - base.x, lm.y - base.y, lm.z - base.z])
        points = np.array(points)
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points /= scale
        sig = points.flatten().tolist()
        self._sig_buffer.append(sig)
        return sig

    def match(self, current_sig, saved_sig):
        """
        Return the maximum cosine similarity between saved_sig and
        any signature in the rolling window (DTW-lite approach).
        Falls back to direct comparison if buffer is empty.
        """
        if current_sig is None or saved_sig is None:
            return 0.0
        s = np.array(saved_sig)
        norms_s = np.linalg.norm(s)
        if norms_s == 0:
            return 0.0

        candidates = list(self._sig_buffer) if self._sig_buffer else [current_sig]
        best = 0.0
        for sig in candidates:
            c = np.array(sig)
            nc = np.linalg.norm(c)
            if nc == 0:
                continue
            score = float(np.dot(c, s) / (nc * norms_s))
            best = max(best, score)
        return best


class ObjectLearner:
    def __init__(self): pass
    def get_hsv_profile(self, frame, roi=None):
        h, w, _ = frame.shape
        if roi is None: roi = [0.4, 0.4, 0.6, 0.6]
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return [int(np.mean(hsv[:,:,0])), int(np.mean(hsv[:,:,1])), int(np.mean(hsv[:,:,2]))]
