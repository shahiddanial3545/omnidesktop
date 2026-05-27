import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMenu, QAction
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QSize
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
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(150, 150, Qt.KeepAspectRatio))

class OmniBubble(QWidget):
    mode_changed = pyqtSignal(str) # "Focus", "Lazy", "Off"

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
        self.bubble.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 150, 255, 180);
                color: white;
                border-radius: 25px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 150, 255, 255);
            }
        """)
        self.bubble.clicked.connect(self.toggle_status)
        self.bubble.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bubble.customContextMenuRequested.connect(self.show_context_menu)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: white; background: rgba(0,0,0,150); border-radius: 5px; padding: 5px;")
        self.status_label.hide()

        self.layout.addWidget(self.bubble)
        self.layout.addWidget(self.status_label)
        self.setLayout(self.layout)

        self.move(100, 100)
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.preview.move(self.x() + 60, self.y())
            self.oldPos = event.globalPos()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("background: rgba(30,30,30,220); color: white; border: 1px solid #555;")

        focus_act = menu.addAction("Focus Mode")
        lazy_act = menu.addAction("Lazy Mode")
        off_act = menu.addAction("All Off")
        menu.addSeparator()
        preview_act = menu.addAction("Toggle Preview")

        action = menu.exec_(self.bubble.mapToGlobal(pos))
        if action == focus_act: self.mode_changed.emit("Focus")
        elif action == lazy_act: self.mode_changed.emit("Lazy")
        elif action == off_act: self.mode_changed.emit("Off")
        elif action == preview_act:
            if self.preview.isVisible(): self.preview.hide()
            else:
                self.preview.move(self.x() + 60, self.y())
                self.preview.show()

    def toggle_status(self):
        if self.status_label.isVisible(): self.status_label.hide()
        else: self.status_label.show()

    def update_status(self, text):
        self.status_label.setText(text)
        QTimer.singleShot(3000, lambda: self.status_label.hide() if self.status_label.text() == text else None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = OmniBubble()
    sys.exit(app.exec_())
