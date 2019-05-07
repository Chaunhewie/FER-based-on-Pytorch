# coding=utf-8
import os
import sys
import traceback
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QApplication, QComboBox
from PyQt5.QtCore import Qt, QRect, QSize

from main_windows.css import *
from main_windows.worker_threads import InitModelThread, FERWorkerThread, VirtualizeWorkerThread

DEBUG = False
ENABLED_DATASET = ['CK+', 'JAFFE', 'FER2013']


class FERWindow(QMainWindow):

    def __init__(self, root_pre_path='main_windows', tr_using_crop=False, *args, **kwargs):
        super(FERWindow, self).__init__(*args, **kwargs)
        self.tr_using_crop = tr_using_crop
        self.n_classes = 7  # 模型输出为7个类
        self.n_features_conv = 9  # 模型由8层卷积层，用于可视化
        self.vir_index = 0  # 用于标识当前识别图片的次数，否则可视化工作线程会覆盖

        model_root_pre_path = ""
        css_root_pre_path = root_pre_path + "/"
        if len(root_pre_path) <= 0:
            model_root_pre_path = ".."
            css_root_pre_path = ""
        self.root_pre_path = root_pre_path
        self.model_root_pre_path = model_root_pre_path
        self.css_root_pre_path = css_root_pre_path

        # 窗口大小
        self.resize(1950, 980)

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
        self.dataset = ENABLED_DATASET[2]
        self.choose_dataset_combobox.setCurrentText(self.dataset)
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

        # 展示模型结构
        self.model_structer_pic_label = QLabel(self)
        self.model_structer_pic_label.setText("正在初始化...")
        self.model_structer_pic_label.setFixedSize(1300, 650)
        self.model_structer_pic_label.move(450, 165)
        self.model_structer_pic_label.setAlignment(Qt.AlignCenter)
        self.model_structer_pic_label.setStyleSheet(SHOW_PIC_LABEL_CSS)
        # 展示模型预测的结果
        self.output_class_labels = []
        self.softmax_rate_labels = []
        self.softmax_rate_fl_labels = []
        self.softmax_rate_real_labels = []
        for i in range(self.n_classes):
            label = QLabel(self)
            label.setText("")
            label.setFixedSize(50, 20)
            label.move(1720, 205+i*50)
            label.setStyleSheet(SHOW_INFO_LABEL_CSS)
            self.output_class_labels.append(label)
            label = QLabel(self)
            label.setText("")
            label.setFixedSize(75, 15)
            label.move(1750, 225+i*50)
            label.setStyleSheet(SHOW_RATE_LABEL_CSS)
            self.softmax_rate_labels.append(label)
            label = QLabel(self)
            label.setText("")
            label.setFixedSize(75, 15)
            label.move(1750, 225+i*50+15)
            label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
            self.softmax_rate_fl_labels.append(label)
            label = QLabel(self)
            label.setText("")
            label.setFixedSize(75, 15)
            label.move(1830, 225+i*50+7)
            label.setStyleSheet(SHOW_RATE_REAL_LABEL_CSS)
            self.softmax_rate_real_labels.append(label)
        # 展示模型的中间输出可视化
        self.virtualizations_name_labels = []
        self.virtualizations_labels = []
        self.virtualizations_fl_name_labels = []
        self.virtualizations_fl_labels = []
        self.virtualizations_x_positions = [450, 575, 700, 825, 950, 1075, 1200, 1325, 1450, 1575]
        for i in range(self.n_features_conv):
            label = QLabel(self)
            label.setText("conv"+str(i+1))
            label.setFixedSize(100, 20)
            label.move(self.virtualizations_x_positions[i], 30)
            label.setStyleSheet(SHOW_RATE_LABEL_CSS)
            self.virtualizations_name_labels.append(label)
            label = QLabel(self)
            label.setText("可视化图片")
            label.setFixedSize(100, 100)
            label.move(self.virtualizations_x_positions[i], 50)
            label.setStyleSheet(SHOW_RATE_LABEL_CSS)
            self.virtualizations_labels.append(label)
            label = QLabel(self)
            label.setText("conv"+str(i+1))
            label.setFixedSize(100, 20)
            label.move(self.virtualizations_x_positions[i], 820)
            label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
            self.virtualizations_fl_name_labels.append(label)
            label = QLabel(self)
            label.setText("可视化图片")
            label.setFixedSize(100, 100)
            label.move(self.virtualizations_x_positions[i], 840)
            label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
            self.virtualizations_fl_labels.append(label)
        # 其他的变量声明初始化
        self.model_controller = None
        self.img_model_struct_path = os.path.join(self.root_pre_path, "Resources", "ACCNN_model_structer.png")
        self.fer_worker_thread = None
        self.virtualize_worker_threads = []
        self.virtualize_fl_worker_threads = []
        for i in range(self.n_features_conv):
            self.virtualize_worker_threads.append(None)
            self.virtualize_fl_worker_threads.append(None)

        self.setWindowTitle("My FER Program")
        self.showMaximized()
        self.show()
        QApplication.processEvents()

        # 加载模型参数
        self.init_model_thread = InitModelThread(model_root_pre_path, dataset="FER2013",
                                                 tr_using_crop=self.tr_using_crop)  # 工作的线程
        self.init_model_thread.signal.connect(self.init_load_model_slot)
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
            self.model_structer_pic_label.width() / self.model_structer_pic_label.height():
            img_model_struct = img_model_struct.scaledToWidth(self.model_structer_pic_label.width())
        else:
            img_model_struct = img_model_struct.scaledToHeight(self.model_structer_pic_label.height())
        self.model_structer_pic_label.setPixmap(img_model_struct)
        for i in range(self.n_classes):
            self.output_class_labels[i].setText(self.model_controller.model.output_map[i])
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
        for i in range(self.n_features_conv):
            self.virtualizations_labels[i].clear()
            self.virtualizations_labels[i].setText("可视化图片")
            self.virtualizations_fl_labels[i].clear()
            self.virtualizations_fl_labels[i].setText("可视化图片")
        QApplication.processEvents()

        # 调用人脸识别的线程进行工作
        self.vir_index += 1
        self.fer_worker_thread = FERWorkerThread(self.model_controller, img_origin, self.vir_index)
        self.fer_worker_thread.signal.connect(self.fer_worker_thread_slot)
        self.fer_worker_thread.start()

    def fer_worker_thread_slot(self, img_origin, res, duration, vir_index):
        """
        人脸识别线程的返回槽
        :param img_origin: 原图
        :param res: 结果(face_box, emotion)
        :param duration: 通过时间
        :return: 无
        """
        if vir_index != self.vir_index:
            return
        try:
            face_box, emotion, softmax_rate, virtualizations = res
            # print(len(virtualizations), len(virtualizations[0]), len(virtualizations[1]))
            # 输入打开的图片到模型中识别并将结果展示
            self.show_emotion_label.setText("预测表情：" + emotion)
            self.show_delay_label.setText("通过时间：" + str(duration) + "ms")
            if DEBUG:
                self.show_debug_label.setText("人脸框：" + str(face_box) + " 图像大小：" + str(img_origin.size()))

            # 绘制人脸定位图像
            img_with_face_box = self.draw_img_with_face_box(img_origin.copy(), face_box)
            self.show_res_label.setText("")
            self.show_res_label.setPixmap(img_with_face_box)
            QApplication.processEvents()

            # 更新神经元节点的QLabel展示值
            for i in range(self.n_classes):
                self.softmax_rate_labels[i].setText("%.2e" % softmax_rate[0][i])
                self.softmax_rate_fl_labels[i].setText("%.2e" % softmax_rate[1][i])
                self.softmax_rate_real_labels[i].setText("%.2e" % softmax_rate[2][i])
                QApplication.processEvents()

            # 更新可视化中间层图像
            for i in range(self.n_features_conv):
                images, images_fl = virtualizations[0][i], virtualizations[1][i]
                img_save_dir = os.path.join(self.model_root_pre_path, "Saved_Virtualizations")

                self.virtualize_worker_threads[i] = VirtualizeWorkerThread(images, "ACCNN_" + self.dataset,
                                                                           img_save_dir, i, False, vir_index)
                self.virtualize_worker_threads[i].signal.connect(self.virtualize_worker_thread_slot)
                self.virtualize_worker_threads[i].start()

                self.virtualize_fl_worker_threads[i] = VirtualizeWorkerThread(images_fl, "ACCNN_fl_" + self.dataset,
                                                                              img_save_dir, i, True, vir_index)
                self.virtualize_fl_worker_threads[i].signal.connect(self.virtualize_worker_thread_slot)
                self.virtualize_fl_worker_threads[i].start()
        except:
            traceback.print_exc()

    def virtualize_worker_thread_slot(self, img_saved_path, i, is_fl, vir_index):
        if vir_index != self.vir_index:
            return
        try:
            if is_fl:
                vir_label = self.virtualizations_fl_labels[i]
            else:
                vir_label = self.virtualizations_labels[i]

            img = QPixmap(img_saved_path)
            if img.width() > img.height():
                img = img.scaledToWidth(self.virtualizations_labels[i].width())
            else:
                img = img.scaledToHeight(self.virtualizations_labels[i].height())
            vir_label.setPixmap(img)
            QApplication.processEvents()
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
        self.dataset = dataset
        self.init_model_thread = InitModelThread(self.model_root_pre_path, dataset=dataset,
                                                 tr_using_crop=self.tr_using_crop)  # 工作的线程
        self.init_model_thread.signal.connect(self.init_load_model_slot)
        self.init_model_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")
    window = FERWindow(root_pre_path='')
    app.exec_()
