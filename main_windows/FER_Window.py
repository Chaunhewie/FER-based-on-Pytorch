# coding=utf-8
import sys
import time
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QApplication, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QRect, QSize

from main_windows.model_controller import ModelController
from main_windows.css import *

QPixmap_Channels_Count = 4


class FERWindow(QMainWindow):

    def __init__(self, root_pre_path='main_windows', *args, **kwargs):
        super(FERWindow, self).__init__(*args, **kwargs)
        model_root_pre_path = ""
        css_root_pre_path = root_pre_path + "/"
        if len(root_pre_path) <= 0:
            model_root_pre_path = ".."
            css_root_pre_path = ""

        # 窗口大小
        self.resize(1000, 1200)

        # 打开图片的btn
        self.get_pic_btn = QPushButton(self)
        self.get_pic_btn.setText("打开图片")
        self.get_pic_btn.setFixedSize(QSize(150, 50))
        self.get_pic_btn.move(20, 20)
        self.get_pic_btn.clicked.connect(self.open_image)
        self.get_pic_btn.setStyleSheet(BUTTON_CSS % (css_root_pre_path, css_root_pre_path))
        # 展示原图片的label
        self.show_pic_label = QLabel(self)
        self.show_pic_label.setText("显示图片")
        self.show_pic_label.setFixedSize(450, 450)
        self.show_pic_label.move(20, 90)
        self.show_pic_label.setStyleSheet(SHOW_PIC_LABEL_CSS)
        # 展示人脸定位的label
        self.show_res_label = QLabel(self)
        self.show_res_label.setText("显示图片")
        self.show_res_label.setFixedSize(450, 450)
        self.show_res_label.move(520, 90)
        self.show_res_label.setStyleSheet(SHOW_PIC_LABEL_CSS)
        # 展示表情的label
        self.show_emotion_label = QLabel(self)
        self.show_emotion_label.setText("")
        self.show_emotion_label.setFixedSize(350, 20)
        self.show_emotion_label.move(520, 50)
        self.show_emotion_label.setStyleSheet(SHOW_INFO_LABEL_CSS)
        # 展示时间的label
        self.show_delay_label = QLabel(self)
        self.show_delay_label.setText("")
        self.show_delay_label.setFixedSize(350, 20)
        self.show_delay_label.move(520, 70)
        self.show_delay_label.setStyleSheet(SHOW_INFO_LABEL_CSS)
        # 展示debug信息的label
        self.show_debug_label = QLabel(self)
        self.show_debug_label.setText("")
        self.show_debug_label.setFixedSize(960, 10)
        self.show_debug_label.move(20, 540)
        self.show_debug_label.setStyleSheet(SHOW_DEBUG_LABEL_CSS)

        # 展示模型可视化结果
        self.virtualize_box = QVBoxLayout(self)
        img_virt_box = QHBoxLayout(self)
        # virt_num =
        img_fl_virt_box = QHBoxLayout(self)
        self.virtualize_box.addItem(img_virt_box)
        self.virtualize_box.addItem(img_fl_virt_box)

        self.setWindowTitle("My FER Program")
        self.show()

        self.model_controller = ModelController(model_root_pre_path=model_root_pre_path)


    def open_image(self):
        # 打开并展示一张图片
        img_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;*.jpg;;All Files(*)")
        if len(img_path) <= 0:
            return
        print("get image path: ", img_path)
        img_origin = QPixmap(img_path)
        if img_origin.width() > img_origin.height():
            img_origin = img_origin.scaledToWidth(self.show_pic_label.width())
        else:
            img_origin = img_origin.scaledToHeight(self.show_pic_label.height())
        self.show_pic_label.setPixmap(img_origin)

        # 输入打开的图片到模型中识别并将结果展示
        h, w = img_origin.height(), img_origin.width()
        img_str = img_origin.toImage().bits().asstring(w * h * QPixmap_Channels_Count)
        img_arr = np.fromstring(img_str, dtype=np.uint8).reshape((h, w, QPixmap_Channels_Count))
        start_time = time.time()
        face_box, emotion = self.model_controller.fer_recognization(img_arr, resize_size=350)
        # print(img_res)
        end_time = time.time()
        duration = round((end_time - start_time) * 1000, 2)
        self.show_emotion_label.setText("预测表情：" + emotion)
        self.show_delay_label.setText("通过时间：" + str(duration) + "ms")
        self.show_debug_label.setText("人脸框：" + str(face_box) + " 图像大小：" + str(img_origin.size()))

        # 绘制人脸定位图像
        img_with_face_box = self.draw_img_with_face_box(img_origin.copy(), face_box)
        self.show_res_label.setPixmap(img_with_face_box)

    def draw_img_with_face_box(self, img, face_box):
        # h, w = img.height(), img.width()
        top, right, bottom, left = face_box
        painter = QPainter(img)
        pen = QPen(Qt.blue, 5)
        painter.setPen(pen)
        painter.begin(img)
        painter.drawRect(QRect(left, top, right-left, bottom-top))  # x, y, width, height
        painter.end()
        return img


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")

    window = FERWindow(root_pre_path='')
    app.exec_()