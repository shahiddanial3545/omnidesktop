import cv2
import numpy as np
import time
import sys

# Extremely resilient MediaPipe loader
def load_mp_solution(name):
    # Try multiple common import paths for MediaPipe solutions
    paths = [
        f"mediapipe.python.solutions.{name}",
        f"mediapipe.solutions.{name}",
        f"solutions.{name}"
    ]

    # Try direct imports first
    for path in paths:
        try:
            __import__(path)
            return sys.modules[path]
        except ImportError:
            continue

    # Try attribute access on mediapipe
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            solutions = getattr(mp, 'solutions')
            if hasattr(solutions, name):
                return getattr(solutions, name)
    except (ImportError, AttributeError):
        pass

    # Try importing solutions subpackage directly
    try:
        import mediapipe.solutions as mp_solutions
        if hasattr(mp_solutions, name):
            return getattr(mp_solutions, name)
    except (ImportError, AttributeError):
        pass

    raise ImportError(f"Could not load MediaPipe solution: {name}. Please ensure mediapipe is installed correctly.")

# Pre-load solutions
try:
    mp_hands = load_mp_solution('hands')
    mp_pose = load_mp_solution('pose')
    mp_face_detection = load_mp_solution('face_detection')
    mp_face_mesh = load_mp_solution('face_mesh')
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    # Define dummy classes to avoid NameError if pre-loading fails but app continues
    class Dummy:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    mp_hands = mp_pose = mp_face_detection = mp_face_mesh = Dummy()

class VisionCore:
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height

        # Lazy initialization to save memory if features are disabled
        self._hands = None
        self._pose = None
        self._face_detection = None
        self._face_mesh = None

    @property
    def hands(self):
        if self._hands is None:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._hands

    @property
    def pose(self):
        if self._pose is None:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._pose

    @property
    def face_detection(self):
        if self._face_detection is None:
            self._face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        return self._face_detection

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._face_mesh

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip for natural interaction
        frame = cv2.flip(frame, 1)
        return frame

    def process(self, frame, features=None):
        if features is None:
            features = ['hands', 'pose', 'face_detection', 'face_mesh']

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}

        if 'hands' in features:
            results['hands'] = self.hands.process(rgb_frame)
        if 'pose' in features:
            results['pose'] = self.pose.process(rgb_frame)
        if 'face_detection' in features:
            results['face_detection'] = self.face_detection.process(rgb_frame)
        if 'face_mesh' in features:
            results['face_mesh'] = self.face_mesh.process(rgb_frame)

        return results

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    vc = VisionCore()
    start_time = time.time()
    frames = 0
    while frames < 10:
        frame = vc.get_frame()
        if frame is not None:
            res = vc.process(frame)
            frames += 1
            print(f"Processed frame {frames}")
    vc.release()
    print(f"FPS: {frames / (time.time() - start_time)}")
