from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2, time, sys
import numpy as np
from playsound import playsound
import face_recognition
from tensorflow import keras
import psutil
import socket
import json
import mediapipe as mp

def getCPU():
    return psutil.cpu_percent(interval=1)

def getRAM():
    return psutil.virtual_memory()

eye_model = keras.models.load_model('yuiyongmodel.h5')

class ThreadClass(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FPS = pyqtSignal(int)
    EyeCrimination = pyqtSignal(int)

    def run(self):
        Capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.ThreadActive = True
        prev_frame_time = 0
        new_frame_time = 0
        while self.ThreadActive:
            ret, frame = Capture.read()
            flip_frame = cv2.flip(src=frame, flipCode= 1)
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            if ret:

                image_for_prediction = self.eye_cropper(frame)
                try:
                    image_for_prediction = image_for_prediction/255.0
                except:
                    continue

                # 모델에서 예측 가져오기.
                prediction = eye_model.predict(image_for_prediction)
                if prediction < 0.5:
                    counter = 0
                    status = 'Open'
                    self.EyeCrimination.emit(counter)
                    
                else:
                    counter = counter + 1
                    status = 'Closed'
                    self.EyeCrimination.emit(counter)

                    # 카운터가 3보다 크면 사용자가 잠들었다는 경고를 재생하고 표시한다.
                    if counter > 3:
                        # 소리 재생
                        playsound('test6.mp3')
                        self.EyeCrimination.emit(counter)
                        counter = 0
                        continue

                self.ImageUpdate.emit(flip_frame)
                self.FPS.emit(fps)
    
    # 웹캠 프레임이 함수에 입력된다.
    def eye_cropper(self, frame):

        # 얼굴 특징 좌표에 대한 변수 생성
        facial_features_list = face_recognition.face_landmarks(frame)
        # print('1번째 테스트',facial_features_list)

        # 눈 좌표에 대한 자리 표시 목록을 만듦.
        try:
            eye = facial_features_list[0]['left_eye']
            # print('2번째 테스트',eye)
        except:
            try:
                eye = facial_features_list[0]['right_eye']
                # print('3번째 테스트',eye)
            except:
                # 안면 인식으로 찾지 못한 경우
                return
        
        # 눈의 최대 x및 y 좌표 설정
        x_max = max([coordinate[0] for coordinate in eye])
        # print(x_max)
        x_min = min([coordinate[0] for coordinate in eye])
        # print(x_min)
        y_max = max([coordinate[1] for coordinate in eye])
        y_min = min([coordinate[1] for coordinate in eye])

        # x와 y 좌표의 범위 설정
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 전체 눈이 캡처되었는지 확인하기 위해
        # 사각형의 좌표를 계산한다.
        # 더 넓은 범위로 x축에 50% 추가
        # 그런 다음 더 작은 범위를 완충된 더 큰 범위에 일치시키는 작업을 수행해야 한다.
        if x_range > y_range:
            right = round(.5*x_range) + x_max
            left = x_min - round(.5*x_range)
            bottom = round((((right-left) - y_range))/2) + y_max
            top = y_min - round((((right-left) - y_range))/2)
        else:
            bottom = round(.5*y_range) + y_max
            top = y_min - round(.5*y_range)
            right = round((((bottom-top) - x_range))/2) + x_max
            left = x_min - round((((bottom-top) - x_range))/2)

        # 위에서 결정한 좌표에 따라 이미지 자르기
        cropped = frame[top:(bottom + 1), left:(right + 1)]

        # 이미지 크기 조정
        cropped = cv2.resize(cropped, (80,80))
        image_for_prediction = cropped.reshape(-1, 80, 80, 3)

        return image_for_prediction

    def stop(self):
        self.ThreadActive = False
        self.quit()

class boardInfo(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:

            cpu = getCPU()
            ram = getRAM()

            self.cpu.emit(cpu)
            self.ram.emit(ram)
    
    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("untitled.ui", self)

        self.myipaddress = socket.gethostbyname(socket.gethostname())
        self.serverconnectionbtn.clicked.connect(self.request)

        self.online_cam = QCameraInfo.availableCameras()
        self.camlistcb.addItems([c.description() for c in self.online_cam])

        self.startcambtn.clicked.connect(self.StartWebCam)
        self.stopcambtn.clicked.connect(self.StopWebCam)
        self.stopcambtn.setEnabled(False)
        self.closeconnectionbtn.setEnabled(False)

        self.resource_usage = boardInfo()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.getCPU_usage)
        self.resource_usage.ram.connect(self.getRAM_usage)

        self.ready_lamp = QTimer(self, interval=1000)
        self.ready_lamp.timeout.connect(self.Ready_lamp)
        self.ready_lamp.start()

        self.motor_on = QTimer(self, interval=1000)
        self.motor_on.timeout.connect(self.Running_lamp)
        self.motor_on.timeout.connect(self.Motor_state)

        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.clock)
        self.lcd_timer.start()

        self.flag_motor = True
        self.Status_lamp = [True, True]
        self.Status_Eye = [True, True, True]
        
        self.cammotorlabel.setPixmap(QPixmap('offmonitor.png'))
        self.closebtn.clicked.connect(self.Close_software)

    def __del__(self):
        self.client_socket.close()

    def Ready_lamp(self):
        if self.Status_lamp[0]:
            self.readylabel.setStyleSheet("background-color: rgb(85, 255, 0); border-radius:30px")
        else:
            self.readylabel.setStyleSheet("background-color: rgb(184, 230, 191); border-radius:30px")
        
        self.Status_lamp[0] = not self.Status_lamp[0]

    def Running_lamp(self):
        if self.Status_lamp[1]:
            self.runninglabel.setStyleSheet("background-color: rgb(255, 195, 0); border-radius:30px")
        else:
            self.runninglabel.setStyleSheet("background-color: rgb(242, 214, 117); border-radius:30px")
        
        self.Status_lamp[1] = not self.Status_lamp[1]

    def request(self):
        self.ip = self.server_addresslineEdit.text()
        self.port = self.portlineEdit.text()
        print(self.myipaddress)
        if len(self.ip) ==0 or len(self.port) == 0:
            QMessageBox.information(self, '주소확인', 'IP 혹은 Port번호를 확인해주세요.')
        else:
            try:
                print("확인중입니다..")
                print(self.ip)
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print(self.client_socket)
                self.client_socket.connect((self.ip, int(self.port)))
                print(type(self.ip), type(self.port))
                self.closeconnectionbtn.setEnabled(True)
                message = {'signal' : f'{self.productareacb.currentText()}', 'IP':f'{self.myipaddress}' }
                message_json = json.dumps(message)
                self.client_socket.sendall(message_json.encode('utf-8'))
                self.productareacb.setEnabled(False)
                self.serverconnectionbtn.setEnabled(False)
                self.connlabel.setStyleSheet("background-color: rgb(85, 255, 0); border-radius:30px")
                print(self.myipaddress)
                print(message_json)
            except:
                QMessageBox.information(self, '연결확인', '연결이 거부되었습니다.')

    def Motor_state(self):
        if self.flag_motor:
            self.cammotorlabel.setPixmap(QPixmap('onmonitor.png'))
        else:
            self.cammotorlabel.setPixmap(QPixmap('offmonitor.png'))
        self.flag_motor = not self.flag_motor

    def getCPU_usage(self,cpu):
        self.CPUlabel.setText(str(cpu) + " %")
        if cpu > 15: self.CPUlabel.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25: self.CPUlabel.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45: self.CPUlabel.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65: self.CPUlabel.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85: self.CPUlabel.setStyleSheet("color: rgb(237, 85, 59);")

    def getRAM_usage(self, ram):
        self.RAMlabel.setText(str(ram[2]) + " %")
        if ram[2] > 15: self.RAMlabel.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25: self.RAMlabel.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45: self.RAMlabel.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65: self.RAMlabel.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85: self.RAMlabel.setStyleSheet("color: rgb(237, 85, 59);")

    def opencv_emit(self, Image):

        original = self.cvt_opencv_qt(Image)
        self.CopyImage = Image[20:2000,
                            20:2000]
        
        self.mainwebcam.setPixmap(original)
        self.mainwebcam.setScaledContents(True)
    
    def cvt_opencv_qt(self, Image):

        rgb_img = cv2.cvtColor(src=Image, code=cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_img.shape
        bytes_per_line = ch * w
        cvt2QtFormat = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(cvt2QtFormat)

        return pixmap

    def get_FPS(self, fps):
        self.fpslabel.setText(str(fps))
        if fps > 5: self.fpslabel.setStyleSheet("color: rgb(237, 85, 59);")
        if fps > 15: self.fpslabel.setStyleSheet("color: rgb(60, 174, 155);")
        if fps > 25: self.fpslabel.setStyleSheet("color: rgb(85, 170, 255);")
        if fps > 35: self.fpslabel.setStyleSheet("color: rgb(23, 63, 95);")

    def clock(self):
        self.DateTime = QDateTime.currentDateTime()
        self.timelcd.display(self.DateTime.toString('hh:mm:ss'))

    def get_drowsiness(self, counter):
        if counter == 0:
            if self.Status_Eye[0]:
                self.openeyelabel.setStyleSheet("background-color: rgb(85, 255, 0); border-radius:30px")
            else:
                self.openeyelabel.setStyleSheet("background-color: rgb(184, 230, 191); border-radius:30px")
            self.Status_Eye[0] = not self.Status_Eye[0]

        if counter > 0 and counter< 4 :
            if self.Status_Eye[1]:
                self.closeeyelabel.setStyleSheet("background-color: rgb(255, 0, 0); border-radius:30px")
            else:
                self.closeeyelabel.setStyleSheet("background-color:  rgb(255, 171, 175); border-radius:30px")
            self.Status_Eye[1] = not self.Status_Eye[1]

        if counter > 3 :
            if self.Status_Eye[2]:
                self.drowsinesslabel.setStyleSheet("background-color: rgb(255, 0, 0); border-radius:30px")  
                self.drowsiness_count += 1 
            else:
                self.drowsinesslabel.setStyleSheet("background-color:  rgb(255, 171, 175); border-radius:30px")
            self.Status_Eye[2] = not self.Status_Eye[2]
            self.drowsinesslcd.display(self.drowsiness_count)

    def StartWebCam(self):

        try:
            self.statusTextEdit.append(f"{self.DateTime.toString('yy년 MMM월 d일 hh:mm:ss')}: Start Webcam ({self.camlistcb.currentText()})")
            self.startcambtn.setEnabled(False)
            self.stopcambtn.setEnabled(True)

            self.Worker_Opencv = ThreadClass()
            self.Worker_Opencv.ImageUpdate.connect(self.opencv_emit)
            self.Worker_Opencv.FPS.connect(self.get_FPS)
            self.Worker_Opencv.EyeCrimination.connect(self.get_drowsiness)
            self.Worker_Opencv.start()
            self.motor_on.start()
            self.drowsiness_count = 0
            self.camlistcb.setEnabled(False)
            self.productareacb.setEnabled(False)
            self.ready_lamp.stop()

        except Exception as error :
            pass
    
    def StopWebCam(self):
        try:
            self.statusTextEdit.append(f"{self.DateTime.toString('yy년 MMM월 d일 hh:mm:ss')}: Stop Webcam ({self.camlistcb.currentText()})")
            self.startcambtn.setEnabled(True)
            self.stopcambtn.setEnabled(False)

            self.cammotorlabel.setPixmap(QPixmap('offmonitor.png'))
            self.motor_on.stop()
            self.Worker_Opencv.stop()
            self.camlistcb.setEnabled(True)
            self.ready_lamp.start()
            self.productareacb.setEnabled(True)

        except Exception as error :
            pass

    def Close_software(self):
        self.Worker_Opencv.stop()
        self.resource_usage.stop()
        self.lcd_timer.stop()
        self.ready_lamp.stop()
        sys.exit(app.exec_())       

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()