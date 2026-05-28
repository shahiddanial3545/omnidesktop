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

# Robust Dummy Class to prevent NoneType attribute errors
class MediaPipeDummy:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name): return self
    def process(self, *args, **kwargs):
        class DummyResults:
            def __init__(self):
                self.multi_hand_landmarks = None
                self.pose_landmarks = None
                self.detections = None
                self.multi_face_landmarks = None
        return DummyResults()

try:
    mp_hands = load_mp_solution('hands')
    mp_pose = load_mp_solution('pose')
    mp_face_detection = load_mp_solution('face_detection')
    mp_face_mesh = load_mp_solution('face_mesh')
except ImportError:
    print("WARNING: MediaPipe could not be loaded. Running in dummy mode.")
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

    def _get_filter(self, key):
        if key not in self.filters:
            self.filters[key] = EMAFilter(self.alpha)
        return self.filters[key]

    @property
    def hands(self):
        if self._hands is None:
            try:
                self._hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except:
                self._hands = MediaPipeDummy()
        return self._hands

    @property
    def pose(self):
        if self._pose is None:
            try:
                self._pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except:
                self._pose = MediaPipeDummy()
        return self._pose

    @property
    def face_detection(self):
        if self._face_detection is None:
            try:
                self._face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            except:
                self._face_detection = MediaPipeDummy()
        return self._face_detection

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            try:
                self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except:
                self._face_mesh = MediaPipeDummy()
        return self._face_mesh

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return cv2.flip(frame, 1)

    def smooth_landmarks(self, results, feature_type):
        if not results: return
        if feature_type == 'hands' and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for idx, lm in enumerate(hand_landmarks.landmark):
                    key = f"hand_{hand_id}_{idx}"
                    lm.x = self._get_filter(key + "_x").apply(lm.x)
                    lm.y = self._get_filter(key + "_y").apply(lm.y)
                    lm.z = self._get_filter(key + "_z").apply(lm.z)
        elif feature_type == 'pose' and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                key = f"pose_{idx}"
                lm.x = self._get_filter(key + "_x").apply(lm.x)
                lm.y = self._get_filter(key + "_y").apply(lm.y)
                lm.z = self._get_filter(key + "_z").apply(lm.z)
        elif feature_type == 'face_mesh' and hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for idx, lm in enumerate(face_landmarks.landmark):
                    key = f"face_{face_id}_{idx}"
                    lm.x = self._get_filter(key + "_x").apply(lm.x)
                    lm.y = self._get_filter(key + "_y").apply(lm.y)
                    lm.z = self._get_filter(key + "_z").apply(lm.z)

    def process(self, frame, features=None):
        if features is None: features = ['hands', 'pose', 'face_detection', 'face_mesh']
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}

        # Safe processing calls
        if 'hands' in features:
            h = self.hands
            if h:
                res = h.process(rgb_frame)
                self.smooth_landmarks(res, 'hands')
                results['hands'] = res
        if 'pose' in features:
            p = self.pose
            if p:
                res = p.process(rgb_frame)
                self.smooth_landmarks(res, 'pose')
                results['pose'] = res
        if 'face_detection' in features:
            fd = self.face_detection
            if fd:
                results['face_detection'] = fd.process(rgb_frame)
        if 'face_mesh' in features:
            fm = self.face_mesh
            if fm:
                res = fm.process(rgb_frame)
                self.smooth_landmarks(res, 'face_mesh')
                results['face_mesh'] = res
        return results

    def release(self):
        self.cap.release()
