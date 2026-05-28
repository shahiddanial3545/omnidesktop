# Project Omni-Desk: The Universal Physical AI Layer

Omni-Desk transforms a standard laptop and webcam into a spatial computer using Edge AI. It bridges your physical world (atoms) and digital world (pixels).

## 🚀 UX 2.0 (Intuitive & Interactive)
The latest version includes **Cross-Platform Audio Feedback** (Speech + Chimes) and a **Visual Preview** system.

### 1. Prerequisites
- Python 3.8+
- Webcam

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe pyautogui PyQt5 numpy screen-brightness-control psutil pyttsx3
```

### 3. Run the Application
```bash
python main.py
```

## 🎮 How to Use (Simplified UX)

### The Floating Bubble
A small blue bubble appears on your screen. 
- **Left-Click:** Move the bubble anywhere.
- **Right-Click:** Open the **Quick Mode Menu**:
  - **🎯 Focus Mode:** Enables Privacy Shield, Posture Guardian, Phone-Down, and Gaze Guard.
  - **😴 Lazy Mode:** Enables Air-Scroll, Paper Dashboard, Coffee Mug Mute, and Palm Menu.
  - **⚡ All On:** Enables every single feature at once.
  - **⏸ All Off:** Pauses processing to save CPU/Battery.
  - **🤙 Record Gesture:** Record a custom hand/pose movement.
  - **📦 Learn Object:** Teach the app to recognize a physical object by its color.
  - **👁 Toggle Preview:** Opens a window showing exactly what the camera sees.

### Audio Feedback (Cross-Platform)
The app communicates using chimes (Windows) and Voice (Speech fallback for all platforms).
- "Dashboard connected" when the camera sees your A4 paper.
- "Muting microphone" when you tap a paper button or shush.
- "Please fix your posture" when you slouch.

## 🛠 Features Breakdown

1.  **Any-Gesture Macro Recorder:** Record ANY movement (skeletal topology) and link it to an OS command or URL.
2.  **Palm Menu:** Pinch your thumb to your fingers to trigger instant actions (like Ctrl+T or Mute).
3.  **Hold-to-Learn Objects:** Hold an object in the center frame to teach the app its HSV profile and assign a macro.
4.  **Auto Paper Dashboard:** Place a white A4 paper on your desk. Tap it to trigger macros.
5.  **Shush Trigger:** Finger to lips instantly mutes volume and hides windows.
6.  **Gaze-Based Dimming:** Screen dims to 10% when you look away and restores instantly when you look back.
7.  **Table Double-Tap:** Double-tap your desk with your hand to Play/Pause (Space).

## 🔒 Security & Privacy
1.  **100% Offline:** No images or data ever leave your machine.
2.  **Privacy Shield:** Built-in protection against shoulder-surfers.

## 🩺 Troubleshooting
If MediaPipe fails or the app crashes, run:
```bash
python diagnose.py
```
To fix your environment automatically:
```bash
pip install --force-reinstall numpy==1.26.4 mediapipe==0.10.13 pyttsx3
```
