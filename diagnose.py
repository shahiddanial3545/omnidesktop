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
            ret, frame = cap.read()
            if ret:
                print(f"Camera Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print("Camera: NOT DETECTED (Check if another app is using it)")
    except ImportError:
        print("OpenCV: NOT INSTALLED")

    print("\n--- MediaPipe Check ---")
    try:
        import mediapipe as mp
        print(f"MediaPipe Version: {mp.__version__}")
        from mediapipe.python.solutions import hands
        try:
            h = hands.Hands(static_image_mode=True)
            print("MediaPipe Initialization: SUCCESS")
        except Exception as e:
            print(f"MediaPipe Initialization: FAILED ({e})")
    except ImportError as e:
        print(f"MediaPipe: NOT ACCESSIBLE ({e})")

    print("\n--- Other Dependencies ---")
    for pkg, import_name in [
        ("PyQt5", "PyQt5"),
        ("pyautogui", "pyautogui"),
        ("screen_brightness_control", "screen_brightness_control"),
        ("pyttsx3", "pyttsx3"),
        ("psutil", "psutil"),
    ]:
        try:
            __import__(import_name)
            print(f"{pkg}: OK")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED  →  pip install {pkg}")

    print("\n--- Fix Command ---")
    print("pip install --force-reinstall numpy==1.26.4 mediapipe==0.10.13 pyttsx3")

if __name__ == "__main__":
    run_diagnostics()
