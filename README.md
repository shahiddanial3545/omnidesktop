# Project Omni-Desk: The Universal Physical AI Layer

Omni-Desk transforms a standard laptop and webcam into a spatial computer using Edge AI. It bridges your physical world (atoms) and digital world (pixels).

## 🚀 NEW: UX 2.0 (Intuitive & Interactive)
The latest version includes **Offline Voice Feedback** and a **Visual Preview** system to make spatial computing effortless.

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
  - **Focus Mode:** Enables Privacy Shield (monitors backgrounds), Posture Guardian (straightens your back), and Phone-Down Focus.
  - **Lazy Mode:** Enables Air-Scroll (scroll hands-free), Paper Dashboard, and Coffee Mug Mute.
  - **All Off:** Pauses processing to save CPU/Battery.
  - **Toggle Preview:** Opens a tiny 150x150 window showing exactly what the camera sees (landmarks included).

### Voice Feedback (Local & Offline)
The app speaks to you! You will hear:
- "Dashboard connected" when the camera sees your A4 paper.
- "Muting microphone" when you tap a paper button.
- "Please fix your posture" when you slouch.

## 🛠 Features Breakdown

1.  **Auto Paper Dashboard:** Place a white A4 paper on your desk. The app detects it automatically. Tap areas on the paper to trigger macros.
2.  **Shush Trigger:** Place your index finger on your lips to instantly mute all volume and minimize non-work windows.
3.  **Table Double-Tap:** Double-tap your desk surface with your hand to Play/Pause media (Spacebar).
4.  **Air-Scroll:** Move your wrist up/down in front of the camera to scroll through TikTok, YouTube, or articles.
5.  **Privacy Shield:** Screen minimizes automatically if someone stands behind you.
6.  **Posture Guardian:** Screen dims if you sit with bad posture for over 10 minutes.

## 🔒 Security & Privacy
1.  **100% Offline:** No images or data ever leave your machine.
2.  **Privacy Shield:** Built-in protection against shoulder-surfers.

## 🖥 Hardware Target
Optimized for **Intel Core i5 (7th Gen)** with < 150MB RAM usage.
