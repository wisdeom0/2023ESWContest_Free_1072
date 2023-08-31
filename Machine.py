from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QDialog, QVBoxLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
import sys
import os
from For_Project import weight_save, Button, Pose_Correct
import csv
import cv2
from multiprocessing import Process, Pipe
imgpath_4 = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\배경4.png"
Exercise = "Lat Pull Down"
next_path  = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\다음버튼.png"
reset_path = r"C:\Users\wcutiy\OneDrive\사진\프로젝트사진\종료버튼.png"
check_path = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\확인버튼.png"
correct_path = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\교정.png"

class UserInputWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('사용자 정보 입력')
        self.user_input = QLineEdit(self)

        self.ok_button = QPushButton('확인', self)
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('사용자의 정보를 입력하세요'))
        layout.addWidget(self.user_input)
        layout.addWidget(self.ok_button)

    def accept(self):
        self.user_data = self.user_input.text()
        super().accept()

csv_route = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진"
def run_another_task(pipe_conn):
    print("Another task is running...")
    while True:
        if pipe_conn.poll():
            frame = pipe_conn.recv()
            exe_gui = '랫풀 다운'
        Pose_Correct.pose_correction(exe_gui, csv_route, pipe_conn)
        pass


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.correction_labels = []
        self.initUserInput()
        self.initUI()
        self.parent_conn, self.child_conn = Pipe()
        self.process = Process(target=run_another_task, args=(self.child_conn,))
        self.process.start()


        self.initCamera()

    def initUserInput(self):
        self.user_input_window = UserInputWindow(self)
        if self.user_input_window.exec_():
            self.user_info = self.user_input_window.user_data
            if not self.user_info == None:
                self.last_info = weight_save.last_row(self.user_info, Exercise)
                with open(self.user_info + "_routine.csv", mode='r', newline='') as f:
                    reader = csv.reader(f)
                    self.routine_names = list(reader)
                    self.routine_names = self.routine_names[0]
                    print(self.routine_names)
            else:
                self.routine_names = None

    def initUI(self):
        print(self.user_info)
        image = QImage()
        image.load(imgpath_4)
        pixmap = QPixmap.fromImage(image)
        self.image_path = imgpath_4
        self.font = QFont()
        self.font.setPointSize(14)
        self.resize(pixmap.width(), pixmap.height())
        self.show_weight()
        screen = QDesktopWidget().screenGeometry()

        self.setGeometry(0, 0, screen.width(), screen.height())

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
        for i, routine in enumerate(self.routine_names):
            frame = Button.ColorFrame(self, 'lightblue', 70, 720 + 100*i, 500, 80, )
            label = Button.Set_Label(self, 90, 760 + 100*i, self.font, routine)
        self.label = Button.Set_Label(self, 910, 40, self.font, "횟수")
        self.label = Button.Set_Label(self, 1415, 40, self.font, "무게")
        self.label = Button.Set_Label(self, 2130, 120, self.font, f"{self.user_info}")
        self.reps_label = Button.Set_Spinbox(self, 910, 90, 0, 200, 1, self.font, 400, 80)
        self.weight_label = Button.Set_Spinbox(self, 1415, 90, 0, 200, 5, self.font, 400, 80)

        self.next_button = Button.HoverButton(
            '다음', self, 2000, 470, 160, 100, 'lightblue', 'gray', 'black', 'black')
        self.end_button = Button.HoverButton(
            '종료', self, 2000, 600, 160, 100, 'pink', 'gray', 'black', 'black')
        self.correction_box = Button.ColorFrame(self, 'lightblue', 740, 1200, 1700, 250)

        self.next_button.clicked.connect(self.next_set_clicked)
        self.end_button.clicked.connect(self.reset)
        self.pose_correction()
        self.setWindowTitle('사용자의 정보를 입력하세요')
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(740, 385, 1000, 780)

        self.show()

        self.pose_correction_timer = QTimer(self)
        self.pose_correction_timer.timeout.connect(self.pose_correction)
        self.pose_correction_timer.start(30)
    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.updateCameraView)
        self.camera_timer.start(30)

    def updateCameraView(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (1000, 780))
            self.parent_conn.send(frame)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.camera_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print('Mouse Position:', event.x(), event.y())

    def reset(self):
        weight_save.input_and_save(self.user_info, Exercise,
            self.weight_label.value(), self.reps_label.value())
        try :
            os.remove(csv_route + "jsonoutput.csv")
        except :
            pass
        self.close()
        self.new_window = MainWindow()
        self.new_window.show()

    def next_set_clicked(self):
        weight_save.input_and_save(self.user_info, Exercise,
            self.weight_label.value(), self.reps_label.value())
        self.weight_label.setValue(0)
        self.reps_label.setValue(0)
        try :
            os.remove(csv_route + "\jsonoutput.csv")
        except :
            pass
        self.show_weight()

    def recommending_weight(self):
        filename = f"{self.user_info}.csv"
        if not os.path.exists(filename):
            return [5, 5, 5]
        else :
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                for row in reversed(data):
                    if row[1] == Exercise:
                        return row[-3:]
            return [5, 5, 5]

    def show_weight(self):
        self.machine_name = Button.Set_Label(self, 110, 330, self.font, Exercise)
        rm = self.recommending_weight()
        self.rm_1 = Button.Set_Label(self, 110, 440, self.font, f"1회 반복 무게 :{rm[0]}")
        self.rm_5 = Button.Set_Label(self, 110, 500, self.font, f"7~10회 반복 무게 :{rm[1]}")
        self.rm_10 = Button.Set_Label(self, 110, 560, self.font, f"11~15회 반복 무게 :{rm[2]}")

    def pose_correction(self):
        for label in self.correction_labels:
            label.deleteLater()

        self.correction_labels.clear()

        correction_list = []
        try:
            with open(csv_route + '\jsonoutput.csv', 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    correction_list = row[:-1]
        except:
            correction_list = None

        if correction_list is not None:
            for i, correction in enumerate(correction_list):
                label = QLabel(correction, self)
                label.setFont(self.font)
                if i % 2:
                    label.move(1250, 1250 + (i - 1) * 25)
                else:
                    label.move(800, 1250 + i * 25)
                label.show()
                self.correction_labels.append(label)
        else:
            label = QLabel("Wait...", self)
            label.setFont(self.font)
            label.move(800, 1260)
            label.show()
            self.correction_labels.append(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())