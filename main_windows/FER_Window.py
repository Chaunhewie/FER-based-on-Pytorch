# coding=utf-8
import os
import sys
import traceback
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog, QApplication, QComboBox
from PyQt5.QtCore import Qt, QRect, QSize, pyqtSignal

from main_windows.css import *
from main_windows.worker_threads import InitModelThread, FERWorkerThread, VirtualizeWorkerThread, VirtualizeBarDistributeThread

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
        self.img_save_dir = os.path.join(self.model_root_pre_path, "Saved_Virtualizations")

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
        self.vir_name_labels = []
        self.vir_labels = []
        self.vir_fl_name_labels = []
        self.vir_fl_labels = []
        self.vir_x_positions = [450, 575, 700, 825, 950, 1075, 1200, 1325, 1450, 1575]
        for i in range(self.n_features_conv):
            label = QLabel(self)
            label.setText("conv"+str(i+1))
            label.setFixedSize(100, 20)
            label.move(self.vir_x_positions[i], 30)
            label.setStyleSheet(SHOW_RATE_LABEL_CSS)
            self.vir_name_labels.append(label)
            label = QLabel(self)
            label.setText("可视化图片")
            label.setFixedSize(100, 100)
            label.move(self.vir_x_positions[i], 50)
            label.setStyleSheet(SHOW_RATE_LABEL_CSS)
            self.vir_labels.append(label)
            label = QLabel(self)
            label.setText("conv"+str(i+1))
            label.setFixedSize(100, 20)
            label.move(self.vir_x_positions[i], 820)
            label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
            self.vir_fl_name_labels.append(label)
            label = QLabel(self)
            label.setText("可视化图片")
            label.setFixedSize(100, 100)
            label.move(self.vir_x_positions[i], 840)
            label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
            self.vir_fl_labels.append(label)
        # 展示输出结果的条形图
        self.vir_softmax_rate_label = QLabel(self)
        self.vir_softmax_rate_label.setText("结果")
        self.vir_softmax_rate_label.setFixedSize(150, 150)
        self.vir_softmax_rate_label.move(1575, 10)
        self.vir_softmax_rate_label.setStyleSheet(SHOW_RATE_LABEL_CSS)
        self.vir_fl_softmax_rate_label = QLabel(self)
        self.vir_fl_softmax_rate_label.setText("结果")
        self.vir_fl_softmax_rate_label.setFixedSize(150, 150)
        self.vir_fl_softmax_rate_label.move(1575, 820)
        self.vir_fl_softmax_rate_label.setStyleSheet(SHOW_RATE_FL_LABEL_CSS)
        # 其他的变量声明初始化
        self.img_model_struct_path = os.path.join(self.root_pre_path, "Resources", "ACCNN_model_structer.png")
        self.model_controller = None  # 用于控制模型操作
        self.fer_worker_thread = None  # 用于人脸识别
        self.vir_worker_threads = []  # 用于绘制图片识别层的中间层输出
        self.vir_fl_worker_threads = []  # 用于绘制关键标记点识别层的中间层输出
        for i in range(self.n_features_conv):
            self.vir_worker_threads.append(None)
            self.vir_fl_worker_threads.append(None)
        self.vir_bar_distribute_thread, self.vir_fl_bar_distribute_thread = None, None  # 用于绘制两层输出的各类分布的条形图

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
        # 停止正在工作的可视化线程
        self.stop_fer_vir_threads()
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
            self.vir_labels[i].clear()
            self.vir_labels[i].setText("可视化图片")
            self.vir_fl_labels[i].clear()
            self.vir_fl_labels[i].setText("可视化图片")
        self.vir_softmax_rate_label.clear()
        self.vir_softmax_rate_label.setText("结果")
        self.vir_fl_softmax_rate_label.clear()
        self.vir_fl_softmax_rate_label.setText("结果")
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
            face_box, emotion, softmax_rates, vir = res
            # print(len(vir), len(vir[0]), len(vir[1]))
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
                self.softmax_rate_labels[i].setText("%.2e" % softmax_rates[0][i])
                self.softmax_rate_fl_labels[i].setText("%.2e" % softmax_rates[1][i])
                self.softmax_rate_real_labels[i].setText("%.2e" % softmax_rates[2][i])
                QApplication.processEvents()

            # 更新可视化中间层图像
            for i in range(self.n_features_conv):
                images, images_fl = vir[0][i], vir[1][i]

                self.vir_worker_threads[i] = VirtualizeWorkerThread(images, "ACCNN_" + self.dataset,
                                                                    self.img_save_dir, i, False, vir_index)
                self.vir_worker_threads[i].signal.connect(self.vir_worker_thread_slot)
                self.vir_worker_threads[i].start()

                self.vir_fl_worker_threads[i] = VirtualizeWorkerThread(images_fl, "ACCNN_" + self.dataset,
                                                                       self.img_save_dir, i, True, vir_index)
                self.vir_fl_worker_threads[i].signal.connect(self.vir_worker_thread_slot)
                self.vir_fl_worker_threads[i].start()

            # 绘制条形图
            self.vir_bar_distribute_thread = VirtualizeBarDistributeThread(self.model_controller.model.output_map,
                                softmax_rates[0], "ACCNN_" + self.dataset, self.img_save_dir, False, vir_index)
            self.vir_bar_distribute_thread.signal.connect(self.vir_bar_worker_thread_slot)
            self.vir_bar_distribute_thread.start()

            self.vir_fl_bar_distribute_thread = VirtualizeBarDistributeThread(self.model_controller.model.output_map,
                                softmax_rates[1], "ACCNN_" + self.dataset, self.img_save_dir, True, vir_index)
            self.vir_fl_bar_distribute_thread.signal.connect(self.vir_bar_worker_thread_slot)
            self.vir_fl_bar_distribute_thread.start()
        except:
            traceback.print_exc()

    def vir_worker_thread_slot(self, img_saved_path, i, is_fl, vir_index):
        if vir_index != self.vir_index:
            return
        try:
            if is_fl:
                self.set_img_to_label(QPixmap(img_saved_path), self.vir_fl_labels[i])
            else:
                self.set_img_to_label(QPixmap(img_saved_path), self.vir_labels[i])
        except:
            traceback.print_exc()

    def vir_bar_worker_thread_slot(self, img_saved_path, is_fl, vir_index):
        if vir_index != self.vir_index:
            return
        try:
            if is_fl:
                self.set_img_to_label(QPixmap(img_saved_path), self.vir_fl_softmax_rate_label)
            else:
                self.set_img_to_label(QPixmap(img_saved_path), self.vir_softmax_rate_label)
        except:
            traceback.print_exc()

    def set_img_to_label(self, img, vir_label):
        if img.width() > img.height():
            img = img.scaledToWidth(vir_label.width())
        else:
            img = img.scaledToHeight(vir_label.height())
        vir_label.setPixmap(img)
        QApplication.processEvents()

    def draw_img_with_face_box(self, img, face_box):
        """
        用于在img上面绘制脸部框图，返回绘制后的图片
        :param img: QPixmap
        :param face_box: (top, right, bottom, left) 脸部框图像素点的位置
        :return: QPixmap
        """
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

    def stop_fer_vir_threads(self):
        num = 0
        if self.fer_worker_thread and self.fer_worker_thread.isRunning():
            num += 1
            self.fer_worker_thread.stop()
        for i in range(self.n_features_conv):
            if self.vir_worker_threads[i] and self.vir_worker_threads[i].isRunning():
                num += 1
                self.vir_worker_threads[i].stop()
            if self.vir_fl_worker_threads[i] and self.vir_worker_threads[i].isRunning():
                num += 1
                self.vir_worker_threads[i].stop()
        if self.vir_bar_distribute_thread and self.vir_bar_distribute_thread.isRunning():
            num += 1
            self.vir_bar_distribute_thread.stop()
        if self.vir_fl_bar_distribute_thread and self.vir_fl_bar_distribute_thread.isRunning():
            num += 1
            self.vir_fl_bar_distribute_thread.stop()
        print("Stopped thread num : %d" % num)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")
    window = FERWindow(root_pre_path='')
    app.exec_()
