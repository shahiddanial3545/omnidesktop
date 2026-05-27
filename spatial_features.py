import cv2
import numpy as np
import time

class PaperDashboard:
    def __init__(self, corners=None, buttons=None):
        self.corners = corners # List of 4 points [(x,y), ...]
        self.buttons = buttons or []
        self.matrix = None
        if self.corners and len(self.corners) == 4:
            self._update_matrix()

    def _update_matrix(self):
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        pts = np.array(self.corners, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Target A4 aspect ratio or just 1000x1000 for simplicity
        dst = np.array([
            [0, 0],
            [1000, 0],
            [1000, 1000],
            [0, 1000]], dtype="float32")

        self.matrix = cv2.getPerspectiveTransform(rect, dst)

    def set_corners(self, corners):
        self.corners = corners
        self._update_matrix()

    def map_point(self, x, y):
        if self.matrix is None:
            return None
        point = np.array([[[x, y]]], dtype="float32")
        transformed = cv2.perspectiveTransform(point, self.matrix)
        return transformed[0][0]

    def check_tap(self, finger_tip):
        if self.matrix is None or finger_tip is None:
            return None

        mapped = self.map_point(finger_tip[0], finger_tip[1])
        if mapped is None:
            return None

        mx, my = mapped[0] / 1000.0, mapped[1] / 1000.0 # Normalize to 0-1

        for btn in self.buttons:
            bx, by, bw, bh = btn['rect']
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                return btn['macro']
        return None

class GhostActions:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.recorded_motions = [] # List of (signature, macro)
        self.prev_frame = None

    def get_signature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        self.prev_frame = gray

        # Resize to a small matrix for fast comparison
        signature = cv2.resize(thresh, (10, 10))
        return signature

    def record_action(self, frames, macro):
        signatures = [self.get_signature(f) for f in frames]
        signatures = [s for s in signatures if s is not None]
        if signatures:
            # Simple average or sequence of signatures
            avg_signature = np.mean(signatures, axis=0).astype(np.uint8)
            self.recorded_motions.append((avg_signature, macro))

    def match_motion(self, current_signature):
        if current_signature is None:
            return None

        # Normalize current signature to handle binary comparison better
        current_signature = (current_signature > 0).astype(np.uint8)

        for signature, macro in self.recorded_motions:
            # Normalize recorded signature
            signature_norm = (signature > 0).astype(np.uint8)

            # Intersection over Union (IoU) or simply Hamming distance
            # Here using percentage of matching pixels
            matches = np.sum(current_signature == signature_norm)
            match_percent = matches / float(current_signature.size)

            if match_percent > 0.8: # 80% match threshold
                return macro
        return None
