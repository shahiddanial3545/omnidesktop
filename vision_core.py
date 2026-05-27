import cv2
import mediapipe as mp
import numpy as np
import time

# Multi-tiered MediaPipe solution imports for maximum compatibility
try:
    # Tier 1: Try importing from the python subpackage
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import face_detection as mp_face_detection
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except ImportError:
    try:
        # Tier 2: Try direct attribute access via solutions submodule
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.pose as mp_pose
        import mediapipe.solutions.face_detection as mp_face_detection
        import mediapipe.solutions.face_mesh as mp_face_mesh
    except ImportError:
        # Tier 3: Final fallback to attribute access on the main package
        # This sometimes fails if not explicitly imported, handled via try-except
        try:
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
        except AttributeError:
            # Last resort: Try importing them individually
            import mediapipe.solutions as solutions
            mp_hands = solutions.hands
            mp_pose = solutions.pose
            mp_face_detection = solutions.face_detection
            mp_face_mesh = solutions.face_mesh

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
    def mp_hands(self):
        if self._hands is None:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1, # Reduced for memory
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._hands

    @property
    def mp_pose(self):
        if self._pose is None:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._pose

    @property
    def mp_face_detection(self):
        if self._face_detection is None:
            self._face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        return self._face_detection

    @property
    def mp_face_mesh(self):
        if self._face_mesh is None:
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False, # Disable for memory
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
        # features: list of ['hands', 'pose', 'face_detection', 'face_mesh']
        if features is None:
            features = ['hands', 'pose', 'face_detection', 'face_mesh']

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = {}

        if 'hands' in features:
            results['hands'] = self.mp_hands.process(rgb_frame)
        if 'pose' in features:
            results['pose'] = self.mp_pose.process(rgb_frame)
        if 'face_detection' in features:
            results['face_detection'] = self.mp_face_detection.process(rgb_frame)
        if 'face_mesh' in features:
            results['face_mesh'] = self.mp_face_mesh.process(rgb_frame)

        return results

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    # Sanity check
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
