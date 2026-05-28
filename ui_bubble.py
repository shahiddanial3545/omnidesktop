import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMenu, QAction, QInputDialog, QLineEdit
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QSize, QMetaObject
from PyQt5.QtGui import QImage, QPixmap

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
    update_status_signal = pyqtSignal(str)
    update_preview_signal = pyqtSignal(object)
    request_input_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.oldPos = self.pos()
        self.preview = PreviewWindow()

    def initUI(self):
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
            self.oldPos = event.globalPos()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("background: rgba(30,30,30,220); color: white; border: 1px solid #555;")
        f_act = menu.addAction("🎯 Focus Mode"); l_act = menu.addAction("😴 Lazy Mode")
        a_act = menu.addAction("⚡ All On"); o_act = menu.addAction("⏸ All Off")
        menu.addSeparator()
        rec_g_act = menu.addAction("🤙 Record Gesture"); lr_o_act = menu.addAction("📦 Learn Object")
        menu.addSeparator()
        p_act = menu.addAction("👁 Toggle Preview")

        action = menu.exec_(self.bubble.mapToGlobal(pos))
        if action == f_act: self.mode_changed.emit("Focus")
        elif action == l_act: self.mode_changed.emit("Lazy")
        elif action == a_act: self.mode_changed.emit("All")
        elif action == o_act: self.mode_changed.emit("Off")
        elif action == rec_g_act: self.record_gesture.emit()
        elif action == lr_o_act: self.learn_object.emit()
        elif action == p_act:
            if self.preview.isVisible(): self.preview.hide()
            else: self.preview.move(self.x() + 60, self.y()); self.preview.show()

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
