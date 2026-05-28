import cv2
import numpy as np
import time
import sys
from collections import deque

# Extremely resilient MediaPipe loader
def load_mp_solution(name):
    paths = [
        f"mediapipe.python.solutions.{name}",
        f"mediapipe.solutions.{name}",
        f"solutions.{name}"
    ]
    for path in paths:
        try:
            __import__(path)
            return sys.modules[path]
        except ImportError:
            continue
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            solutions = getattr(mp, 'solutions')
            if hasattr(solutions, name):
                return getattr(solutions, name)
    except (ImportError, AttributeError):
        pass
    try:
        import mediapipe.solutions as mp_solutions
        if hasattr(mp_solutions, name):
            return getattr(mp_solutions, name)
    except (ImportError, AttributeError):
        pass
    raise ImportError(f"Could not load MediaPipe solution: {name}.")

# Robust Dummy Classes
class MediaPipeResults:
    def __init__(self):
        self.multi_hand_landmarks = None
        self.pose_landmarks = None
        self.detections = None
        self.multi_face_landmarks = None

class MediaPipeDummy:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name):
        if name in ["Hands", "Pose", "FaceDetection", "FaceMesh"]:
            return MediaPipeDummy
        return self
    def process(self, *args, **kwargs):
        return MediaPipeResults()

try:
    mp_hands = load_mp_solution('hands')
    mp_pose = load_mp_solution('pose')
    mp_face_detection = load_mp_solution('face_detection')
    mp_face_mesh = load_mp_solution('face_mesh')
except Exception as e:
    print(f"WARNING: MediaPipe could not be loaded ({e}). Running in dummy mode.")
    mp_hands = mp_pose = mp_face_detection = mp_face_mesh = MediaPipeDummy()


class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def apply(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


# ============================================================
# PILLAR 3 — GAZE DIRECTION ESTIMATOR
# ============================================================

class GazeEstimator:
    """
    PILLAR 3: Compute a normalized 2-D gaze vector from FaceMesh iris
    landmarks (refined landmarks required: refine_landmarks=True).

    Iris landmark indices (MediaPipe FaceMesh with refine_landmarks):
      Left eye iris centre  : 468
      Right eye iris centre : 473
      Left eye outer corner : 33   inner corner : 133
      Right eye outer corner: 263  inner corner : 362

    We estimate horizontal and vertical gaze offset by comparing the
    iris center position relative to the eye bounding box.  Returns a
    (gaze_x, gaze_y) tuple in [-1, 1] space:
      gaze_x  > 0  → looking right
      gaze_y  > 0  → looking down
    is_on_screen() returns True when gaze is within the central zone,
    meaning the user is actually looking at the monitor.
    """
    # Iris / eye corner indices
    LEFT_IRIS  = 468
    RIGHT_IRIS = 473
    L_OUTER, L_INNER = 33, 133
    R_OUTER, R_INNER = 263, 362
    L_TOP, L_BOT     = 159, 145   # eyelid (blink too)
    R_TOP, R_BOT     = 386, 374

    def __init__(self, screen_zone=0.35, smoothing=0.4):
        self._screen_zone = screen_zone   # fraction of normalised range
        self._ema_x = EMAFilter(smoothing)
        self._ema_y = EMAFilter(smoothing)

    def estimate(self, face_mesh_results):
        """
        Returns (gaze_x, gaze_y, is_on_screen, blink_dist).
        Returns (0, 0, True, 1.0) when no face is detected.
        """
        if (face_mesh_results is None
                or not hasattr(face_mesh_results, 'multi_face_landmarks')
                or not face_mesh_results.multi_face_landmarks):
            return 0.0, 0.0, True, 1.0

        lm = face_mesh_results.multi_face_landmarks[0].landmark

        # ---- Horizontal gaze (iris relative to eye width) ----
        def iris_ratio(iris_idx, outer_idx, inner_idx):
            iris  = np.array([lm[iris_idx].x,  lm[iris_idx].y])
            outer = np.array([lm[outer_idx].x, lm[outer_idx].y])
            inner = np.array([lm[inner_idx].x, lm[inner_idx].y])
            eye_w = np.linalg.norm(outer - inner)
            if eye_w < 1e-6:
                return 0.5
            return float(np.dot(iris - inner, outer - inner) / (eye_w ** 2))

        left_ratio  = iris_ratio(self.LEFT_IRIS,  self.L_OUTER, self.L_INNER)
        right_ratio = iris_ratio(self.RIGHT_IRIS, self.R_OUTER, self.R_INNER)
        h_ratio = (left_ratio + right_ratio) / 2.0   # 0=far left, 1=far right
        gaze_x_raw = (h_ratio - 0.5) * 2.0           # map to [-1, 1]

        # ---- Vertical gaze (iris relative to eye height) ----
        def v_ratio(iris_idx, top_idx, bot_idx):
            iris = np.array([lm[iris_idx].x, lm[iris_idx].y])
            top  = np.array([lm[top_idx].x,  lm[top_idx].y])
            bot  = np.array([lm[bot_idx].x,  lm[bot_idx].y])
            eye_h = np.linalg.norm(bot - top)
            if eye_h < 1e-6:
                return 0.5
            return float(np.linalg.norm(iris - top) / eye_h)

        left_v  = v_ratio(self.LEFT_IRIS,  self.L_TOP, self.L_BOT)
        right_v = v_ratio(self.RIGHT_IRIS, self.R_TOP, self.R_BOT)
        gaze_y_raw = ((left_v + right_v) / 2.0 - 0.5) * 2.0

        # ---- Blink distance (for fatigue index) ----
        blink_dist = float(np.sqrt(
            (lm[self.L_TOP].x - lm[self.L_BOT].x)**2 +
            (lm[self.L_TOP].y - lm[self.L_BOT].y)**2))

        # ---- EMA smoothing ----
        gaze_x = self._ema_x.apply(gaze_x_raw)
        gaze_y = self._ema_y.apply(gaze_y_raw)

        is_on_screen = (abs(gaze_x) < self._screen_zone and
                        abs(gaze_y) < self._screen_zone)

        return gaze_x, gaze_y, is_on_screen, blink_dist


class VisionCore:
    def __init__(self, camera_id=0, width=640, height=480, alpha=0.3):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        self.alpha = alpha

        self._hands = None
        self._pose = None
        self._face_detection = None
        self._face_mesh = None
        self.filters = {}

        # PILLAR 3 — shared gaze estimator accessible to habit_engine
        self.gaze_estimator = GazeEstimator()

    def _get_filter(self, key):
        if key not in self.filters:
            self.filters[key] = EMAFilter(self.alpha)
        return self.filters[key]

    @property
    def hands(self):
        if self._hands is None:
            try:
                if hasattr(mp_hands, 'Hands'):
                    self._hands = mp_hands.Hands(
                        static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else:
                    self._hands = MediaPipeDummy()
            except:
                self._hands = MediaPipeDummy()
        return self._hands

    @property
    def pose(self):
        if self._pose is None:
            try:
                if hasattr(mp_pose, 'Pose'):
                    self._pose = mp_pose.Pose(
                        static_image_mode=False, model_complexity=0,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else:
                    self._pose = MediaPipeDummy()
            except:
                self._pose = MediaPipeDummy()
        return self._pose

    @property
    def face_detection(self):
        if self._face_detection is None:
            try:
                if hasattr(mp_face_detection, 'FaceDetection'):
                    self._face_detection = mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5)
                else:
                    self._face_detection = MediaPipeDummy()
            except:
                self._face_detection = MediaPipeDummy()
        return self._face_detection

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            try:
                if hasattr(mp_face_mesh, 'FaceMesh'):
                    self._face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False, max_num_faces=1,
                        refine_landmarks=True,          # required for iris tracking
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
                else:
                    self._face_mesh = MediaPipeDummy()
            except:
                self._face_mesh = MediaPipeDummy()
        return self._face_mesh

    def is_ready(self):
        return self.cap.isOpened()

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def restart_camera(self, camera_id=0):
        self.cap.release()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def smooth_landmarks(self, results, feature_type):
        if results is None:
            return
        if feature_type == 'hands' and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for idx, lm in enumerate(hand_landmarks.landmark):
                    key = f"hand_{hand_id}_{idx}"
                    lm.x = self._get_filter(key+"_x").apply(lm.x)
                    lm.y = self._get_filter(key+"_y").apply(lm.y)
                    lm.z = self._get_filter(key+"_z").apply(lm.z)
        elif feature_type == 'pose' and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                key = f"pose_{idx}"
                lm.x = self._get_filter(key+"_x").apply(lm.x)
                lm.y = self._get_filter(key+"_y").apply(lm.y)
                lm.z = self._get_filter(key+"_z").apply(lm.z)
        elif feature_type == 'face_mesh' and hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for idx, lm in enumerate(face_landmarks.landmark):
                    key = f"face_{face_id}_{idx}"
                    lm.x = self._get_filter(key+"_x").apply(lm.x)
                    lm.y = self._get_filter(key+"_y").apply(lm.y)
                    lm.z = self._get_filter(key+"_z").apply(lm.z)

    def process(self, frame, features=None):
        if features is None:
            features = ['hands', 'pose', 'face_detection', 'face_mesh']
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        if 'hands' in features:
            res = self.hands.process(rgb_frame)
            self.smooth_landmarks(res, 'hands')
            results['hands'] = res
        if 'pose' in features:
            res = self.pose.process(rgb_frame)
            self.smooth_landmarks(res, 'pose')
            results['pose'] = res
        if 'face_detection' in features:
            results['face_detection'] = self.face_detection.process(rgb_frame)
        if 'face_mesh' in features:
            res = self.face_mesh.process(rgb_frame)
            self.smooth_landmarks(res, 'face_mesh')
            results['face_mesh'] = res
        return results

    def release(self):
        self.cap.release()
