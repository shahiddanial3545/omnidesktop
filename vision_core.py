import cv2
import numpy as np
import time
import sys

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

class TrajectoryTracker:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.smoothed_sig = None

    def update(self, current_sig):
        if self.smoothed_sig is None:
            self.smoothed_sig = np.array(current_sig)
        else:
            self.smoothed_sig = self.alpha * np.array(current_sig) + (1 - self.alpha) * self.smoothed_sig
        return self.smoothed_sig.tolist()

class PointKalmanFilter:
    def __init__(self, process_noise=0.03, measurement_noise=0.5, error_init=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimated_error = error_init
        self.current_estimate = None

    def update(self, measurement):
        if self.current_estimate is None:
            self.current_estimate = measurement
            return measurement

        # Prediction
        prediction = self.current_estimate
        self.estimated_error += self.process_noise

        # Update
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_noise)
        self.current_estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimated_error = (1 - kalman_gain) * self.estimated_error

        return self.current_estimate

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
        self.kalman_filters = {}
        self.gaze_vector = [0.5, 0.5] # Default center

    def get_gaze_vector(self, face_landmarks):
        """Calculates rough gaze vector using eye-to-iris relationship."""
        if not face_landmarks: return [0.5, 0.5]
        lm = face_landmarks.landmark
        # Using landmarks for eyes and irises (approximate)
        # Left eye: 33, 133; Left iris center: 468
        # Right eye: 362, 263; Right iris center: 473
        l_eye_l, l_eye_r = lm[33], lm[133]
        r_eye_l, r_eye_r = lm[362], lm[263]
        l_iris, r_iris = lm[468], lm[473]

        # Calculate horizontal and vertical gaze ratio
        lx = (l_iris.x - l_eye_l.x) / (l_eye_r.x - l_eye_l.x + 1e-6)
        rx = (r_iris.x - r_eye_l.x) / (r_eye_r.x - r_eye_l.x + 1e-6)
        ly = (l_iris.y - (l_eye_l.y + l_eye_r.y)/2) / (abs(l_eye_r.x - l_eye_l.x) + 1e-6)

        gx = (lx + rx) / 2
        gy = ly + 0.5 # Normalizing around 0.5
        self.gaze_vector = [np.clip(gx, 0, 1), np.clip(gy, 0, 1)]
        return self.gaze_vector

    def _get_filter(self, key):
        if key not in self.filters:
            self.filters[key] = EMAFilter(self.alpha)
        return self.filters[key]

    def _get_kalman(self, key):
        if key not in self.kalman_filters:
            self.kalman_filters[key] = PointKalmanFilter()
        return self.kalman_filters[key]

    @property
    def hands(self):
        if self._hands is None:
            try:
                if hasattr(mp_hands, 'Hands'):
                    self._hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else: self._hands = MediaPipeDummy()
            except: self._hands = MediaPipeDummy()
        return self._hands

    @property
    def pose(self):
        if self._pose is None:
            try:
                if hasattr(mp_pose, 'Pose'):
                    self._pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else: self._pose = MediaPipeDummy()
            except: self._pose = MediaPipeDummy()
        return self._pose

    @property
    def face_detection(self):
        if self._face_detection is None:
            try:
                if hasattr(mp_face_detection, 'FaceDetection'):
                    self._face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
                else: self._face_detection = MediaPipeDummy()
            except: self._face_detection = MediaPipeDummy()
        return self._face_detection

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            try:
                if hasattr(mp_face_mesh, 'FaceMesh'):
                    # Enable refine_landmarks for gaze tracking (Pillar 4)
                    self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else: self._face_mesh = MediaPipeDummy()
            except: self._face_mesh = MediaPipeDummy()
        return self._face_mesh

    def is_ready(self):
        return self.cap.isOpened()

    def get_frame(self):
        if not self.cap.isOpened(): return None
        ret, frame = self.cap.read()
        if not ret: return None
        return cv2.flip(frame, 1)

    def restart_camera(self, camera_id=0):
        self.cap.release()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def smooth_landmarks(self, results, feature_type, use_kalman=False):
        if results is None: return
        if feature_type == 'hands' and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for idx, lm in enumerate(hand_landmarks.landmark):
                    key = f"hand_{hand_id}_{idx}"
                    if use_kalman:
                        lm.x = self._get_kalman(key+"_x").update(lm.x)
                        lm.y = self._get_kalman(key+"_y").update(lm.y)
                        lm.z = self._get_kalman(key+"_z").update(lm.z)
                    else:
                        lm.x = self._get_filter(key+"_x").apply(lm.x)
                        lm.y = self._get_filter(key+"_y").apply(lm.y)
                        lm.z = self._get_filter(key+"_z").apply(lm.z)
        elif feature_type == 'pose' and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                key = f"pose_{idx}"; lm.x = self._get_filter(key+"_x").apply(lm.x); lm.y = self._get_filter(key+"_y").apply(lm.y); lm.z = self._get_filter(key+"_z").apply(lm.z)
        elif feature_type == 'face_mesh' and hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for idx, lm in enumerate(face_landmarks.landmark):
                    key = f"face_{face_id}_{idx}"; lm.x = self._get_filter(key+"_x").apply(lm.x); lm.y = self._get_filter(key+"_y").apply(lm.y); lm.z = self._get_filter(key+"_z").apply(lm.z)

    def process(self, frame, features=None, use_kalman=False):
        if features is None: features = ['hands', 'pose', 'face_detection', 'face_mesh']
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}
        if 'hands' in features:
            res = self.hands.process(rgb_frame)
            self.smooth_landmarks(res, 'hands', use_kalman=use_kalman)
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
