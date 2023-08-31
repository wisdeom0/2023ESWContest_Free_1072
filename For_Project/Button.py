from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QSpinBox, QLineEdit, QLabel, QFrame
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QEvent
import sys
next_path  = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\다음버튼.png"
reset_path = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\종료버튼.png"
check_path = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\확인버튼.png"
class HoverButton(QPushButton):
    def __init__(self, title, parent, x, y, w, h, db, hb, df, hf):
        super().__init__(title, parent)
        self.default_style = f"""
            QPushButton {{
                background-color: {db};
                color: {df};
            }}
        """
        self.hover_style = f"""
            QPushButton {{
                background-color: {hb};
                color: {hf};
            }}
        """
        self.setStyleSheet(self.default_style)
        self.setFixedSize(w, h)
        self.move(x, y)

    def event(self, e):
        if e.type() == QEvent.HoverEnter:
            self.setStyleSheet(self.hover_style)
        elif e.type() == QEvent.HoverLeave:
            self.setStyleSheet(self.default_style)
        return super().event(e)

def ImageButton(parent, x, y, w, h, image_path, title=""):
    button = QPushButton(title, parent)
    button_style = f"""
        QPushButton {{
            background-image: url({image_path});
        }}
    """
    button.setStyleSheet(button_style)
    button.setFixedSize(w, h)
    button.move(x, y)
    return button

def ImageFrame(parent, image_path, x, y, w, h):  # 부모 매개변수 추가
    frame = QFrame(parent)

    # QPixmap 객체를 생성하고 이미지 경로를 설정합니다.
    pixmap = QPixmap(image_path)

    # QPalette 객체를 생성하여 위젯의 배경에 이미지를 설정합니다.
    palette = QPalette()
    palette.setBrush(QPalette.Background, QBrush(pixmap))
    frame.setPalette(palette)

    # 위젯의 크기를 이미지의 크기와 일치하도록 설정합니다.
    frame.setGeometry(x, y, w, h)
    frame.setAutoFillBackground(True)

    # 테두리와 스타일을 설정합니다.
    frame.setFrameShape(QFrame.Box)
    frame.setFrameShadow(QFrame.Sunken)

    return frame

def ColorFrame(parent, color, x, y, w, h):
    frame = QFrame(parent)
    frame.setStyleSheet(f"background-color: {color};")
    frame.setGeometry(x, y, w, h)
    return frame

def Set_Label(parent, x, y, font, text) :
    Label = QLabel(parent= parent)
    Label.setText(text)
    Label.setFont(font)
    Label.move(x, y)
    return Label

def Set_Spinbox(parent, x, y, min_val, max_val, step_size, font, width, height):
    spin_box = QSpinBox(parent = parent)
    spin_box.setRange(min_val, max_val)
    spin_box.setFont(font)
    spin_box.setSingleStep(step_size)
    spin_box.move(x, y)
    spin_box.setFixedSize(width, height)
    return spin_box
