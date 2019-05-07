# coding=utf-8
import os
import sys
import traceback
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QApplication, QComboBox
from PyQt5.QtCore import Qt, QRect, QSize

from main_windows.css import *
from main_windows.worker_threads import InitModelThread, FERWorkerThread

DEBUG = False
ENABLED_DATASET = ['CK+', 'JAFFE', 'FER2013']


class FERWindow(QMainWindow):

    def __init__(self, root_pre_path='main_windows', tr_using_crop=True, *args, **kwargs):
        super(FERWindow, self).__init__(*args, **kwargs)
        self.tr_using_crop = tr_using_crop

        model_root_pre_path = ""
        css_root_pre_path = root_pre_path + "/"
        if len(root_pre_path) <= 0:
            model_root_pre_path = ".."
            css_root_pre_path = ""
        self.root_pre_path = root_pre_path
        self.model_root_pre_path = model_root_pre_path
        self.css_root_pre_path = css_root_pre_path

        # 窗口大小
        self.resize(1900, 980)

        # 打开图片的btn
        self.get_pic_btn = QPushButton(self)
        self.get_pic_btn.setText("打开图片")
        self.get_pic_btn.setFixedSize(QSize(160, 60))
        self.get_pic_btn.move(20, 20)
        self.get_pic_btn.clicked.connect(self.open_image)
        self.get_pic_btn.setStyleSheet(BUTTON_CSS % (css_root_pre_path, css_root_pre_path))
        # 切换模型训练参数的选项框
        self.choose_dataset_combobox = QComboBox(self)
        self.choose_dataset_combobox.addItems(ENABLED_DATASET)
        self.choose_dataset_combobox.setCurrentText(ENABLED_DATASET[2])
        self.choose_dataset_combobox.currentIndexChanged.connect(lambda: self.change_dataset(
            self.choose_dataset_combobox.currentText()))
        self.choose_dataset_combobox.setFixedSize(QSize(100, 40))
        self.choose_dataset_combobox.setStyleSheet(COMBOBOX_CSS)
        self.choose_dataset_combobox.move(200, 30)
        # 展示原图片的label
        self.show_pic_label = QLabel(self)
        self.show_pic_label.setText("正在初始化...")
        self.show_pic_label.setFixedSize(400, 400)
        self.show_pic_label.move(20, 100)
        self.show_pic_label.setStyleSheet(SHOW_PIC_LABEL_CSS)
        # 展示人脸定位的label
        self.show_res_label = QLabel(self)
        self.show_res_label.setText("正在初始化...")
        self.show_res_label.setFixedSize(400, 400)
        self.show_res_label.move(20, 550)
        self.show_res_label.setStyleSheet(SHOW_PIC_LABEL_CSS)
        # 展示表情的label
        self.show_emotion_label = QLabel(self)
        self.show_emotion_label.setText("")
        self.show_emotion_label.setFixedSize(350, 20)
        self.show_emotion_label.move(20, 510)
        self.show_emotion_label.setStyleSheet(SHOW_INFO_LABEL_CSS)
        # 展示时间的label
        self.show_delay_label = QLabel(self)
        self.show_delay_label.setText("")
        self.show_delay_label.setFixedSize(350, 20)
        self.show_delay_label.move(20, 530)
        self.show_delay_label.setStyleSheet(SHOW_INFO_LABEL_CSS)
        # 展示debug信息的label
        if DEBUG:
            self.show_debug_label = QLabel(self)
            self.show_debug_label.setText("")
            self.show_debug_label.setFixedSize(1000, 10)
            self.show_debug_label.move(20, 500)
            self.show_debug_label.setStyleSheet(SHOW_DEBUG_LABEL_CSS)

        # 展示模型可视化结果
        self.virtulizing_label = QLabel(self)
        self.virtulizing_label.setText("正在初始化...")
        self.virtulizing_label.setFixedSize(1400, 700)
        self.virtulizing_label.move(450, 100)
        self.virtulizing_label.setAlignment(Qt.AlignCenter)
        self.virtulizing_label.setStyleSheet(SHOW_PIC_LABEL_CSS)

        # 其他的变量声明初始化
        self.model_controller = None
        self.img_model_struct_path = os.path.join(self.root_pre_path, "Resources", "ACCNN_model_structer.png")
        self.fer_worker_thread = None

        self.setWindowTitle("My FER Program")
        self.show()
        QApplication.processEvents()

        # 加载模型参数
        self.init_model_thread = InitModelThread(model_root_pre_path, dataset="FER2013",
                                                 tr_using_crop=self.tr_using_crop)  # 工作的线程
        self.init_model_thread._signal.connect(self.init_load_model_slot)
        self.init_model_thread.start()

    def init_load_model_slot(self, model_controller):
        """
        加载模型参数线程的槽
        :param model_controller: 线程返回的信号，ModelController
        :return: 无
        """
        self.model_controller = model_controller
        self.show_pic_label.setText("显示图片")
        self.show_res_label.setText("显示图片")
        img_model_struct = QPixmap(self.img_model_struct_path)
        if img_model_struct.width() / img_model_struct.height() > \
            self.virtulizing_label.width() / self.virtulizing_label.height():
            img_model_struct = img_model_struct.scaledToWidth(self.virtulizing_label.width())
        else:
            img_model_struct = img_model_struct.scaledToHeight(self.virtulizing_label.height())
        self.virtulizing_label.setPixmap(img_model_struct)
        QApplication.processEvents()

    def open_image(self):
        """
        打开图片，并输入模型进行识别，界面更新处理结果
        :return:
        """
        # 打开并展示一张图片
        img_path, _ = QFileDialog.getOpenFileName(None, "打开图片", "", "*.png;*.jpg;;All Files(*)")
        if len(img_path) <= 0:
            return
        print("get image path: ", img_path)
        img_origin = QPixmap(img_path)
        if img_origin.width() > img_origin.height():
            img_origin = img_origin.scaledToWidth(self.show_pic_label.width())
        else:
            img_origin = img_origin.scaledToHeight(self.show_pic_label.height())
        self.show_pic_label.setPixmap(img_origin)
        self.show_res_label.setText("识别中...")
        QApplication.processEvents()

        # 调用人脸识别的线程进行工作
        self.fer_worker_thread = FERWorkerThread(self.model_controller, img_origin)
        self.fer_worker_thread._signal.connect(self.fer_worker_thread_slot)
        self.fer_worker_thread.start()

    def fer_worker_thread_slot(self, img_origin, res, duration):
        """
        人脸识别线程的返回槽
        :param img_origin: 原图
        :param res: 结果(face_box, emotion)
        :param duration: 通过时间
        :return: 无
        """
        try:
            face_box, emotion, softmax_rate = res
            print("slot:", res)
            # 输入打开的图片到模型中识别并将结果展示
            self.show_emotion_label.setText("预测表情：" + emotion)
            self.show_delay_label.setText("通过时间：" + str(duration) + "ms")
            if DEBUG:
                self.show_debug_label.setText("人脸框：" + str(face_box) + " 图像大小：" + str(img_origin.size()))

            # 绘制人脸定位图像
            img_with_face_box = self.draw_img_with_face_box(img_origin.copy(), face_box)
            self.show_res_label.setText("")
            self.show_res_label.setPixmap(img_with_face_box)
        except:
            traceback.print_exc()

    def draw_img_with_face_box(self, img, face_box):
        """
        用于在img上面绘制脸部框图，返回绘制后的图片
        :param img: QPixmap
        :param face_box: (top, right, bottom, left) 脸部框图像素点的位置
        :return: QPixmap
        """
        # h, w = img.height(), img.width()
        top, right, bottom, left = face_box
        painter = QPainter(img)
        pen = QPen(Qt.blue, 5)
        painter.setPen(pen)
        painter.begin(img)
        painter.drawRect(QRect(left, top, right-left, bottom-top))  # x, y, width, height
        painter.end()
        return img


    def change_dataset(self, dataset):
        """
        更换用于训练模型的dataset
        :param dataset: dataset
        :return: 无
        """
        self.show_pic_label.setText("正在初始化...")
        self.show_res_label.setText("正在初始化...")
        self.init_model_thread = InitModelThread(self.model_root_pre_path, dataset=dataset,
                                                 tr_using_crop=self.tr_using_crop)  # 工作的线程
        self.init_model_thread._signal.connect(self.init_load_model_slot)
        self.init_model_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")
    window = FERWindow(root_pre_path='')
    app.exec_()
