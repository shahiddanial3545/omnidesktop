import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QMenu, QAction, QInputDialog, QLineEdit, QDialog,
                             QFormLayout, QSlider, QCheckBox, QListWidget, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QSize, QMetaObject
from PyQt5.QtGui import QImage, QPixmap, QFont

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Omni-Desk Settings")
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QFormLayout(container)

        self.fps_slider = self._create_slider(5, 60, self.config['system'].get('fps', 30))
        form.addRow("Target FPS:", self.fps_slider)

        self.skip_slider = self._create_slider(0, 10, self.config['system'].get('frame_skip', 0))
        form.addRow("Global Frame Skip:", self.skip_slider)

        # Habit Thresholds
        form.addRow(QLabel("<b>Habit Thresholds</b>"))

        self.posture_timeout = self._create_slider(10, 1200, self.config['habits']['posture_guardian'].get('slouch_timeout', 600))
        form.addRow("Posture Timeout (s):", self.posture_timeout)

        self.privacy_sens = self._create_slider(1, 100, int(self.config['habits']['privacy_shield'].get('sensitivity', 0.5) * 100))
        form.addRow("Privacy Sensitivity:", self.privacy_sens)

        self.shush_dist = self._create_slider(1, 200, int(self.config['habits']['shush_trigger'].get('dist_threshold', 0.05) * 1000))
        form.addRow("Shush Distance:", self.shush_dist)

        self.scroll_sens = self._create_slider(1, 100, int(self.config['habits']['air_scroll'].get('sensitivity', 0.1) * 100))
        form.addRow("Scroll Sensitivity:", self.scroll_sens)

        self.gaze_timeout = self._create_slider(1, 30, self.config['habits']['gaze_dimmer'].get('away_timeout', 5))
        form.addRow("Gaze Dim Timeout (s):", self.gaze_timeout)

        self.dim_level = self._create_slider(0, 100, self.config['habits']['gaze_dimmer'].get('dim_level', 10))
        form.addRow("Dim Level (%):", self.dim_level)

        # Toggles
        form.addRow(QLabel("<b>Toggle Features</b>"))
        self.posture_cb = QCheckBox("Enable Posture Guardian"); self.posture_cb.setChecked(self.config['habits']['posture_guardian'].get('enabled', True))
        form.addRow(self.posture_cb)
        self.privacy_cb = QCheckBox("Enable Privacy Shield"); self.privacy_cb.setChecked(self.config['habits']['privacy_shield'].get('enabled', True))
        form.addRow(self.privacy_cb)
        self.gaze_cb = QCheckBox("Enable Gaze Dimmer"); self.gaze_cb.setChecked(self.config['habits']['gaze_dimmer'].get('enabled', True))
        form.addRow(self.gaze_cb)

        self.doubletap_cb = QCheckBox("Enable Double Tap")
        self.doubletap_cb.setChecked(self.config['habits']['double_tap'].get('enabled', True))
        form.addRow(self.doubletap_cb)

        self.palm_cb = QCheckBox("Enable Palm Menu")
        self.palm_cb.setChecked(self.config['habits']['palm_menu'].get('enabled', True))
        form.addRow(self.palm_cb)

        self.mug_cb = QCheckBox("Enable Coffee Mug Mute")
        self.mug_cb.setChecked(self.config['habits']['coffee_mug_mute'].get('enabled', True))
        form.addRow(self.mug_cb)

        self.phone_cb = QCheckBox("Enable Phone Detector")
        self.phone_cb.setChecked(self.config['habits']['phone_down'].get('enabled', True))
        form.addRow(self.phone_cb)

        self.morning_cb = QCheckBox("Enable Morning Routine")
        self.morning_cb.setChecked(self.config['habits']['morning_routine'].get('enabled', True))
        form.addRow(self.morning_cb)

        self.scroll_cb = QCheckBox("Enable Air Scroll"); self.scroll_cb.setChecked(self.config['habits']['air_scroll'].get('enabled', True))
        form.addRow(self.scroll_cb)

        self.voice_cb = QCheckBox("Enable Voice Commands")
        self.voice_cb.setChecked(self.config.get('system', {}).get('voice_enabled', False))
        form.addRow(self.voice_cb)

        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet("background-color: #2980b9; padding: 10px; border-radius: 5px;")
        save_btn.clicked.connect(self.save_settings)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        layout.addWidget(save_btn)
        self.setLayout(layout)

    def _create_slider(self, min_v, max_v, curr_v):
        s = QSlider(Qt.Horizontal)
        s.setRange(min_v, max_v)
        s.setValue(curr_v)
        return s

    def save_settings(self):
        self.config['system']['fps'] = self.fps_slider.value()
        self.config['system']['frame_skip'] = self.skip_slider.value()
        self.config['habits']['posture_guardian']['enabled'] = self.posture_cb.isChecked()
        self.config['habits']['posture_guardian']['slouch_timeout'] = self.posture_timeout.value()
        self.config['habits']['privacy_shield']['enabled'] = self.privacy_cb.isChecked()
        self.config['habits']['privacy_shield']['sensitivity'] = self.privacy_sens.value() / 100.0
        self.config['habits']['shush_trigger']['dist_threshold'] = self.shush_dist.value() / 1000.0
        self.config['habits']['air_scroll']['enabled'] = self.scroll_cb.isChecked()
        self.config['habits']['air_scroll']['sensitivity'] = self.scroll_sens.value() / 100.0
        self.config['habits']['gaze_dimmer']['enabled'] = self.gaze_cb.isChecked()
        self.config['habits']['gaze_dimmer']['away_timeout'] = self.gaze_timeout.value()
        self.config['habits']['gaze_dimmer']['dim_level'] = self.dim_level.value()

        self.config['habits']['double_tap']['enabled'] = self.doubletap_cb.isChecked()
        self.config['habits']['palm_menu']['enabled'] = self.palm_cb.isChecked()
        self.config['habits']['coffee_mug_mute']['enabled'] = self.mug_cb.isChecked()
        self.config['habits']['phone_down']['enabled'] = self.phone_cb.isChecked()
        self.config['habits']['morning_routine']['enabled'] = self.morning_cb.isChecked()
        self.config.setdefault('system', {})['voice_enabled'] = self.voice_cb.isChecked()

        self.accept()

class ActivityLogDialog(QDialog):
    def __init__(self, log_data, focus_score=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Activity Log")
        self.setFixedSize(400, 500)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        layout = QVBoxLayout()

        if focus_score is not None:
            score_label = QLabel(f"Focus Score: {focus_score}/100")
            score_label.setAlignment(Qt.AlignCenter)
            score_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00adb5; padding: 10px;")
            layout.addWidget(score_label)

        self.list_widget = QListWidget()
        self.list_widget.addItems(log_data[::-1])  # Show latest first
        self.list_widget.setStyleSheet("background-color: #34495e; border: none; padding: 5px;")
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

class CameraErrorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Error")
        self.setFixedSize(300, 150)
        self.setStyleSheet("background-color: #c0392b; color: white;")
        layout = QVBoxLayout()
        label = QLabel("Webcam not detected or disconnected.\nPlease check your connection.")
        label.setAlignment(Qt.AlignCenter)
        retry_btn = QPushButton("Retry Connection")
        retry_btn.setStyleSheet("background-color: #e67e22; padding: 10px;")
        retry_btn.clicked.connect(self.accept)
        layout.addWidget(label)
        layout.addWidget(retry_btn)
        self.setLayout(layout)

class PomodoroTimer(QWidget):
    finished = pyqtSignal()

    def __init__(self, minutes=25):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(120, 40)
        self.minutes = minutes
        self.seconds = 0
        self.layout = QVBoxLayout()
        self.label = QLabel(f"{self.minutes:02d}:{self.seconds:02d}")
        self.label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 18px; background: rgba(0,0,0,150); border-radius: 5px; padding: 2px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.hide()

    def start(self, mins=25):
        self.minutes = mins
        self.seconds = 0
        self.timer.start(1000)
        self.show()

    def stop(self):
        self.timer.stop()
        self.hide()

    def tick(self):
        if self.seconds == 0:
            if self.minutes == 0:
                self.stop()
                self.finished.emit()
                return
            self.minutes -= 1
            self.seconds = 59
        else:
            self.seconds -= 1
        self.label.setText(f"{self.minutes:02d}:{self.seconds:02d}")

class DashboardEditorDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Edit Paper Dashboard")
        self.setFixedSize(500, 400)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        for btn in self.config.get('paper_dashboard', {}).get('buttons', []):
            self.list_widget.addItem(f"{btn['name']} | {btn['rect']} | {btn['macro']}")

        layout.addWidget(QLabel("Current Buttons:"))
        layout.addWidget(self.list_widget)

        form = QFormLayout()
        self.name_input = QLineEdit()
        self.rect_input = QLineEdit()
        self.rect_input.setPlaceholderText("x1, y1, x2, y2 (floats 0.0-1.0)")
        self.macro_input = QLineEdit()
        form.addRow("Name:", self.name_input)
        form.addRow("Rect:", self.rect_input)
        form.addRow("Macro/Path:", self.macro_input)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add/Update")
        add_btn.clicked.connect(self.add_button)
        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self.delete_button)
        save_btn = QPushButton("Save & Close")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(add_btn); btn_layout.addWidget(del_btn); btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def add_button(self):
        try:
            name = self.name_input.text()
            rect = [float(x.strip()) for x in self.rect_input.text().split(',')]
            macro = self.macro_input.text()
            if len(rect) != 4: raise ValueError

            # Update if exists
            buttons = self.config['paper_dashboard'].setdefault('buttons', [])
            for btn in buttons:
                if btn['name'] == name:
                    btn['rect'] = rect; btn['macro'] = macro
                    break
            else:
                buttons.append({"name": name, "rect": rect, "macro": macro})

            self.refresh_list()
        except:
            pass

    def delete_button(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            name = current_item.text().split('|')[0].strip()
            self.config['paper_dashboard']['buttons'] = [b for b in self.config['paper_dashboard']['buttons'] if b['name'] != name]
            self.refresh_list()

    def refresh_list(self):
        self.list_widget.clear()
        for btn in self.config.get('paper_dashboard', {}).get('buttons', []):
            self.list_widget.addItem(f"{btn['name']} | {btn['rect']} | {btn['macro']}")

class OnboardingWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Omni-Desk")
        self.setFixedSize(500, 400)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.step = 0
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Welcome to Project Omni-Desk.\n\nLet's set up your spatial workspace.")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 12))

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)

        self.layout.addWidget(self.label)
        self.layout.addStretch()
        self.layout.addWidget(self.next_btn)
        self.setLayout(self.layout)

    def next_step(self):
        self.step += 1
        if self.step == 1:
            self.label.setText("Step 1: Calibration\n\nSit straight for Posture Guardian calibration. The system will track your shoulder line.")
        elif self.step == 2:
            self.label.setText("Step 2: Paper Dashboard\n\nPlace an A4 sheet on your desk. The webcam will automatically detect it to create your touchable surface.")
        elif self.step == 3:
            self.label.setText("Step 3: Gestures\n\nYou can now record custom gestures and link them to apps or macros.")
        else:
            self.accept()

class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(150, 150)
        self.layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(150, 150)
        self.video_label.setStyleSheet("background: rgba(0,0,0,100); border-radius: 10px;")
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)
        self.hide()

    def update_frame(self, frame):
        if frame is None: return
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(150, 150, Qt.KeepAspectRatio))

class OmniBubble(QWidget):
    mode_changed = pyqtSignal(str)
    record_gesture = pyqtSignal()
    learn_object = pyqtSignal()
    config_updated = pyqtSignal(dict)
    show_log_requested = pyqtSignal()
    update_status_signal = pyqtSignal(str)
    update_preview_signal = pyqtSignal(object)
    show_message_signal = pyqtSignal(str, str, str) # title, message, type (info/warning/error)
    show_stats_requested = pyqtSignal()
    edit_dashboard_requested = pyqtSignal()
    pomodoro_finished = pyqtSignal()
    camera_retry_requested = pyqtSignal()
    request_input_signal = pyqtSignal(str, str)

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.preview = PreviewWindow() # Initialize preview BEFORE initUI
        self.pomodoro = PomodoroTimer()
        self.initUI()
        self.oldPos = self.pos()

    def _handle_pomodoro_finished(self):
        self.update_status("Pomodoro Finished!")
        self.pomodoro_finished.emit()

    def initUI(self):
        self.pomodoro.finished.connect(self._handle_pomodoro_finished)
        self.show_message_signal.connect(self.show_message_box)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout()
        self.bubble = QPushButton("O", self)
        self.bubble.setFixedSize(50, 50)
        self.bubble.setStyleSheet("background-color: rgba(0, 150, 255, 180); color: white; border-radius: 25px; font-size: 20px; font-weight: bold;")
        self.bubble.clicked.connect(self.toggle_status)
        self.bubble.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bubble.customContextMenuRequested.connect(self.show_context_menu)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: white; background: rgba(0,0,0,150); border-radius: 5px; padding: 5px;")
        self.status_label.hide()
        self.layout.addWidget(self.bubble); self.layout.addWidget(self.status_label)
        self.setLayout(self.layout); self.move(100, 100); self.show()

        # Connect internal signals
        self.update_status_signal.connect(self.update_status)
        self.update_preview_signal.connect(self.preview.update_frame)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.preview.move(self.x() + 60, self.y())
            self.pomodoro.move(self.x() - 130, self.y() + 5)
            self.oldPos = event.globalPos()

    def show_context_menu(self, pos):
        self.pomodoro.move(self.x() - 130, self.y() + 5)
        menu = QMenu(self)
        menu.setStyleSheet("background: rgba(30,30,30,220); color: white; border: 1px solid #555;")
        f_act = menu.addAction("🎯 Focus Mode"); l_act = menu.addAction("😴 Lazy Mode")
        a_act = menu.addAction("⚡ All On"); o_act = menu.addAction("⏸ All Off")
        menu.addSeparator()
        rec_g_act = menu.addAction("🤙 Record Gesture"); lr_o_act = menu.addAction("📦 Learn Object")
        menu.addSeparator()
        pom_act = menu.addAction("⏲ Start Pomodoro (25m)")
        menu.addSeparator()
        log_act = menu.addAction("📋 Activity Log"); stats_act = menu.addAction("📊 View Stats Report")
        dash_act = menu.addAction("📋 Edit Dashboard")
        sett_act = menu.addAction("⚙️ Settings"); wiz_act = menu.addAction("🧙 Onboarding Wizard")
        menu.addSeparator()
        p_act = menu.addAction("👁 Toggle Preview")

        action = menu.exec_(self.bubble.mapToGlobal(pos))
        if action == f_act: self.mode_changed.emit("Focus")
        elif action == l_act: self.mode_changed.emit("Lazy")
        elif action == a_act: self.mode_changed.emit("All")
        elif action == o_act: self.mode_changed.emit("Off")
        elif action == rec_g_act: self.record_gesture.emit()
        elif action == lr_o_act: self.learn_object.emit()
        elif action == log_act: self.show_log_requested.emit()
        elif action == stats_act: self.show_stats_requested.emit()
        elif action == dash_act: self.edit_dashboard_requested.emit()
        elif action == pom_act: self.pomodoro.start(25)
        elif action == sett_act: self.open_settings()
        elif action == wiz_act: self.open_wizard()
        elif action == p_act:
            if self.preview.isVisible(): self.preview.hide()
            else: self.preview.move(self.x() + 60, self.y()); self.preview.show()

    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec_():
            self.config_updated.emit(self.config)
            self.update_status("Settings Updated")

    def open_wizard(self):
        wiz = OnboardingWizard(self)
        wiz.exec_()

    def open_dashboard_editor(self):
        dlg = DashboardEditorDialog(self.config, self)
        if dlg.exec_():
            self.config_updated.emit(self.config)

    def show_camera_error(self):
        dlg = CameraErrorDialog(self)
        if dlg.exec_():
            self.camera_retry_requested.emit()

    def show_log(self, log_data, focus_score=None):
        dlg = ActivityLogDialog(log_data, focus_score, self)
        dlg.exec_()

    def show_message_box(self, title, message, msg_type="info"):
        from PyQt5.QtWidgets import QMessageBox
        if msg_type == "info":
            QMessageBox.information(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        elif msg_type == "error":
            QMessageBox.critical(self, title, message)

    def get_macro_input(self, title="Action", label="Command:"):
        text, ok = QInputDialog.getText(self, title, label, QLineEdit.Normal, "")
        return text if ok else None

    def toggle_status(self):
        if self.status_label.isVisible(): self.status_label.hide()
        else: self.status_label.show()

    def update_status(self, text):
        self.status_label.setText(text); self.status_label.show()
        QTimer.singleShot(3000, lambda: self.status_label.hide() if self.status_label.text() == text else None)

if __name__ == "__main__":
    app = QApplication(sys.argv); ex = OmniBubble(); sys.exit(app.exec_())
