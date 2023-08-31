from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import sys
from For_Project import Button, Recommanding, Sensor


imgpath_1 = r'/home/storm/mu_code/stormfile/image/배경1.png'
imgpath_2 = r'/home/storm/mu_code/stormfile/image/배경2.png'
imgpath_3 = r'/home/storm/mu_code/stormfile/image/배경3.png'
imgpath_4 = r'/home/storm/mu_code/stormfile/image/배경4.png'
button_path = r"/home/storm/mu_code/stormfile/image/운동부위 버튼.png"

class WorkerThread(QThread):
    result_signal = pyqtSignal(str)

    def run(self):
        global text
        uid = Sensor.read_rfid()
        text = Sensor.rfid(uid)
        self.result_signal.emit(text)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = imgpath_1
        self.initUI()

    def initUI(self):
        # get the screen resolution
        screen_resolution = QApplication.desktop().screenGeometry()
        # Load the image
        image = QImage(imgpath_1)
        image = image.scaled(1920, 1080)
        # Scale image to fit screen
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(screen_resolution.width(), screen_resolution.height(), Qt.KeepAspectRatio)

        self.resize(screen_resolution.width(), screen_resolution.height())

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
        self.font = QFont()
        self.font.setPointSize(14)
        self.button = Button.HoverButton('확인', self, 910, 650, 105, 45,
                                         'white', 'gray', 'black', 'black')
        self.button.clicked.connect(self.on_button_clicked)  # 버튼 클릭 시그널 연결

        self.lineEdit = QLineEdit(self)
        self.lineEdit.move(830, 420)
        self.lineEdit.setFixedSize(400, 40)
        self.lineEdit.setFont(self.font)
        self.setWindowTitle('사용자의 정보를 입력하세요')
        self.show()

        self.worker_thread = WorkerThread()
        self.worker_thread.result_signal.connect(self.rfid)
        self.worker_thread.start()

    def rfid(self):
        self.lineEdit.setText(f"{text}")

        self.second_window = SecondWindow(text)
        self.second_window.show()
        self.hide()



    def on_button_clicked(self):
        text = self.lineEdit.text()
        self.second_window = SecondWindow(text)
        self.second_window.show()
        self.hide()
        #self.read_rfid_and_show_second_window()

class SecondWindow(QWidget):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.image_path = imgpath_2
        self.initUI()

    def initUI(self):
        self.font = QFont()
        self.font.setPointSize(30)
        screen_resolution = QApplication.desktop().screenGeometry()
        # Load the image
        image = QImage(imgpath_2)
        image = image.scaled(1920, 1080, Qt.IgnoreAspectRatio)

        # Scale image to fit screen

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(image))
        self.setPalette(palette)
        self.setGeometry(0, 0, screen_resolution.width(), screen_resolution.height())
        button_positions = [(273, 172, "Back"), (273, 634, "Shoulder"), (940, 169, "Chest"), (940, 634, "Leg")]
        self.buttons = [Button.ImageButton(self, x, y, 500, 400, button_path, title=f"{z}") for x, y , z in button_positions]
        for button in self.buttons:
            button.setFont(self.font)
            button.clicked.connect(self.on_button_clicked)

        self.setWindowTitle('Second Window')
        self.show()

    def on_button_clicked(self):
        self.exercise_list = ['Back', 'Shoulder', 'Chest', 'Leg']

        clicked_button = self.sender()
        clicked_index = self.buttons.index(clicked_button)
        exercise = self.exercise_list[clicked_index]

        for button in self.buttons:
            if button is clicked_button:
                button.setStyleSheet("background-color: green")
            else:
                button.setStyleSheet("background-color: transparent")

        self.third_window = ThirdWindow(self.text, exercise)
        self.third_window.show()

        # 현재 윈도우 숨기기
        self.hide()

class ThirdWindow(QWidget):
    def __init__(self, user, exercise):
        super().__init__()
        self.user = user
        self.exercise = exercise
        self.image_path = imgpath_3
        self.routine = Recommanding.get_routine(self.user, self.exercise)
        self.initUI()



    def initUI(self):
        screen_resolution = QApplication.desktop().screenGeometry()
        # Load the image
        image = QImage(imgpath_3)
        image = image.scaled(1920, 1080, Qt.IgnoreAspectRatio)
        self.font = QFont()
        self.font.setPointSize(14)
        # Scale image to fit screen

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(image))
        self.setPalette(palette)
        self.setGeometry(0, 0, screen_resolution.width(), screen_resolution.height())

        self.routine = Recommanding.get_routine(self.user, self.exercise)
        self.confirm_button = QPushButton("확인", self)
        self.confirm_button.move(1700, 940)  # x, y 좌표는 필요에 따라 조절하십시오.
        self.confirm_button.setFixedSize(120, 60)
        self.confirm_button.clicked.connect(self.back_to_main)
        self.confirm_button.setFont(self.font)
        self.font.setPointSize(20)
        self.label = QLabel(self.exercise, self)
        self.label.move(1550,75)
        self.label.setFixedSize(100, 25)
        self.label.setFont(self.font)
        for i in range(len(self.routine)):
            frame = Button.ColorFrame(self, 440, 190 + 125 * i, 1380, 110, "lightblue")
            self.exercise_name = QLabel(f"Exercise Name : {self.routine[i]}", self)
            self.exercise_name.move(470, 213 + i * 125)
            self.exercise_name.setFont(self.font)
            lastrow = Recommanding.get_lastest_row(self.user, self.routine[i])
            lastrow = lastrow[-3:]
            self.weight_list = QLabel(f"1RM : {lastrow[0]}kg   5~10RM : {lastrow[1]}kg   11~15RM : {lastrow[2]}kg", self)
            self.weight_list.move(470, 250 + i * 125)
            self.weight_list.setFont(self.font)
            self.exercise_name.show()
            self.weight_list.show

        self.setPalette(palette)
        self.setWindowTitle('Third Window')
        self.show()

    def back_to_main(self):
        self.main_window = MainWindow()  # MainWindow를 새로 생성
        self.main_window.show()  # MainWindow를 보여줌
        self.close()  # 현재 ThirdWindow를 닫음

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print('Mouse Position:', event.x(), event.y())



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())