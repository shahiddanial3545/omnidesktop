import cv2
import json
import numpy as np

def calibrate():
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    cap = cv2.VideoCapture(config['system']['camera_id'])
    corners = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x, y])
            print(f"Corner {len(corners)}: ({x}, {y})")
            if len(corners) == 4:
                print("Calibration Complete!")

    cv2.namedWindow("Calibrate Paper Dashboard")
    cv2.setMouseCallback("Calibrate Paper Dashboard", click_event)

    print("Click the 4 corners of your A4 paper in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw current corners
        for pt in corners:
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)

        if len(corners) == 4:
            cv2.polylines(frame, [np.array(corners)], True, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save or 'r' to reset", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibrate Paper Dashboard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(corners) == 4:
            config['paper_dashboard']['corners'] = corners
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print("Configuration saved!")
            break
        elif key == ord('r'):
            corners = []
        elif key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
