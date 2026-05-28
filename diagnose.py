import sys
import os

def run_diagnostics():
    print("--- Omni-Desk Diagnostics ---")
    print(f"Python Version: {sys.version}")

    try:
        import numpy
        print(f"NumPy Version: {numpy.__version__}")
    except ImportError:
        print("NumPy: NOT INSTALLED")

    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Camera: DETECTED")
            cap.release()
        else:
            print("Camera: NOT DETECTED (Check if another app is using it)")
    except ImportError:
        print("OpenCV: NOT INSTALLED")

    print("\n--- MediaPipe Check ---")
    try:
        import mediapipe as mp
        print(f"MediaPipe Version: {mp.__version__}")
        print(f"MediaPipe Location: {mp.__file__}")

        from mediapipe.python.solutions import hands
        print("MediaPipe Solutions: ACCESSIBLE")

        try:
            h = hands.Hands(static_image_mode=True)
            print("MediaPipe Initialization: SUCCESS")
        except Exception as e:
            print(f"MediaPipe Initialization: FAILED ({type(e).__name__}: {e})")
            if "protobuf" in str(e).lower():
                print("SUGGESTION: Protobuf version conflict detected.")
            if "numpy" in str(e).lower():
                print("SUGGESTION: NumPy 2.x is likely incompatible with this MediaPipe version. Downgrade to NumPy 1.26.4.")

    except ImportError as e:
        print(f"MediaPipe: NOT ACCESSIBLE ({e})")
        print("SUGGESTION: Try 'pip install mediapipe' again.")

    print("\n--- Fix Command ---")
    print("If you see errors above, run this command:")
    print("pip install --force-reinstall numpy==1.26.4 mediapipe==0.10.13")

if __name__ == "__main__":
    run_diagnostics()
