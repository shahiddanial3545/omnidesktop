# Project Omni-Desk: The Universal Physical AI Layer

Omni-Desk transforms a standard laptop and webcam into a spatial computer. It maps your physical desk to digital commands and automates your workflow based on your habits.

## 🚀 Quick Start (How to Run)

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed.

### 2. Install Dependencies
Open your terminal and run:
```bash
pip install opencv-python mediapipe pyautogui PyQt5 numpy screen-brightness-control psutil
```

### 3. Run the Application
Navigate to the project folder and execute:
```bash
python main.py
```

## 🛠 Features

*   **The Paper Dashboard:** Draw buttons on an A4 sheet. The system maps them to OS commands.
*   **Ghost Actions:** Record a motion (like closing a book) and link it to an action (like locking your PC).
*   **Posture Guardian:** Automatically dims the screen if you slouch for too long.
*   **Privacy Shield:** Instantly minimizes all windows if a second person is detected behind you.
*   **Phone-Down Focus:** Triggers "Do Not Disturb" mode when your phone is placed face-down on the desk.
*   **Coffee Mug Mute:** Mutes your microphone automatically when you lift your mug.
*   **Air-Scroll:** Scroll through content by moving your hand in the air.

## ⚙️ Configuration
The `config.json` file allows you to customize sensitivity, macros, and calibration points.

*   `paper_dashboard`: Define button coordinates and actions.
*   `habits`: Enable/Disable specific features and set thresholds.

## 🖥 Hardware Requirements
*   **Processor:** Intel Core i5 (7th Gen or better)
*   **RAM:** < 150MB overhead
*   **Camera:** Standard Integrated Webcam

## 🔒 Privacy
Omni-Desk runs **100% Offline**. No data is sent to the cloud. All vision processing happens locally on your CPU.
