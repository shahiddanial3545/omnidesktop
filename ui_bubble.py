import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QPoint, QTimer

class OmniBubble(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.oldPos = self.pos()

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
        self.bubble.clicked.connect(self.toggle_menu)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: white; background: rgba(0,0,0,100); border-radius: 5px; padding: 2px;")
        self.status_label.hide()

        self.layout.addWidget(self.bubble)
        self.layout.addWidget(self.status_label)
        self.setLayout(self.layout)

        self.move(100, 100)
        self.show()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def toggle_menu(self):
        if self.status_label.isVisible():
            self.status_label.hide()
        else:
            self.status_label.show()

    def update_status(self, text):
        self.status_label.setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = OmniBubble()
    sys.exit(app.exec_())
