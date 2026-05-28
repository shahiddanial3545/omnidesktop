import cv2
import numpy as np
import time

class AutoPaperDetector:
    def __init__(self):
        self.last_corners = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50000: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                self.last_corners = approx.reshape(4, 2).tolist()
                return self.last_corners
        return None

class PaperDashboard:
    def __init__(self, corners=None, buttons=None):
        self.corners = corners
        self.buttons = buttons or []
        self.matrix = None
        if self.corners and len(self.corners) == 4: self._update_matrix()

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
        self._update_matrix()

    def check_tap(self, finger_tip):
        if self.matrix is None or finger_tip is None: return None
        point = np.array([[[finger_tip[0], finger_tip[1]]]], dtype="float32")
        transformed = cv2.perspectiveTransform(point, self.matrix)[0][0]
        mx, my = transformed[0] / 1000.0, transformed[1] / 1000.0
        for btn in self.buttons:
            bx, by, bw, bh = btn['rect']
            if bx <= mx <= bx + bw and by <= my <= by + bh: return btn['macro']
        return None

class SkeletalTopology:
    def __init__(self, tolerance=0.85):
        self.tolerance = tolerance

    def get_signature(self, landmarks):
        if not landmarks: return None
        # Use relative distances from wrist (landmark 0)
        base = landmarks.landmark[0]
        points = []
        for lm in landmarks.landmark[1:]: # Skip wrist
            points.append([lm.x - base.x, lm.y - base.y, lm.z - base.z])
        points = np.array(points)
        # Normalize by scale (max distance from wrist)
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0: points /= scale
        return points.flatten().tolist()

    def match(self, current_sig, saved_sig):
        if current_sig is None or saved_sig is None: return 0
        c = np.array(current_sig); s = np.array(saved_sig)
        # Cosine Similarity
        dot = np.dot(c, s)
        norm_c = np.linalg.norm(c); norm_s = np.linalg.norm(s)
        if norm_c == 0 or norm_s == 0: return 0
        return dot / (norm_c * norm_s)

class ObjectLearner:
    def __init__(self):
        pass

    def get_hsv_profile(self, frame, roi=None):
        # roi: [x1, y1, x2, y2] normalized
        h, w, _ = frame.shape
        if roi is None: roi = [0.4, 0.4, 0.6, 0.6]
        x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Calculate dominant color (mean)
        avg_h = int(np.mean(hsv[:,:,0]))
        avg_s = int(np.mean(hsv[:,:,1]))
        avg_v = int(np.mean(hsv[:,:,2]))
        return [avg_h, avg_s, avg_v]
