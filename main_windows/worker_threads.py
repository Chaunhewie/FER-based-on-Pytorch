# coding=utf-8
import time
import numpy as np
import traceback
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from main_windows.model_controller import ModelController
from virtualize import build_and_draw_img_of_features, build_and_draw_bar_img


# 初始化模型和模型参数的工作线程
class InitModelThread(QThread):
    #  通过类成员对象定义信号对象
    signal = pyqtSignal(ModelController)

    def __init__(self, model_root_pre_path, dataset, tr_using_crop):
        super(InitModelThread, self).__init__()
        self.model_root_pre_path = model_root_pre_path
        self.dataset = dataset
        self.tr_using_crop = tr_using_crop
        self._isRunning = False

    def __del__(self):
        self.wait()

    def run(self):
        self._isRunning = True
        try:
            model_controller = ModelController(model_root_pre_path=self.model_root_pre_path, dataset=self.dataset, tr_using_crop=self.tr_using_crop)
            if self._isRunning:
                self.signal.emit(model_controller)  # 注意这里与_signal = pyqtSignal(str)中的类型相同
        except:
            traceback.print_exc()

    def stop(self):
        self._isRunning = False

    def isRunning(self):
        return self._isRunning


# 人脸识别的工作线程
class FERWorkerThread(QThread):
    #  通过类成员对象定义信号对象
    signal = pyqtSignal(QPixmap, tuple, float, int)

    def __init__(self, model_controller, img_origin, vir_index):
        super(FERWorkerThread, self).__init__()
        self.model_controller = model_controller
        self.img_origin = img_origin
        self.QPixmap_Channels_Count = 4
        self.vir_index = vir_index
        self._isRunning = False

    def __del__(self):
        self.wait()

    def run(self):
        self._isRunning = True
        # 将img_origin转化为np的array
        try:
            h, w = self.img_origin.height(), self.img_origin.width()
            img_str = self.img_origin.toImage().bits().asstring(w * h * self.QPixmap_Channels_Count)
            img_arr = np.fromstring(img_str, dtype=np.uint8).reshape((h, w, self.QPixmap_Channels_Count))
            # 计时并进行表情识别
            start_time = time.time()
            res = self.model_controller.fer_recognization(img_arr)
            end_time = time.time()
            duration = round((end_time - start_time) * 1000, 2)
            # 回传
            if self._isRunning:
                self.signal.emit(self.img_origin, res, duration, self.vir_index)
        except:
            traceback.print_exc()

    def stop(self):
        self._isRunning = False

    def isRunning(self):
        return self._isRunning


# 提取模型中间层输出后的可视化工作线程
class VirtualizeWorkerThread(QThread):
    #  通过类成员对象定义信号对象
    signal = pyqtSignal(str, int, bool, int)

    def __init__(self, images, img_name_pre, img_save_dir, index, is_fl, vir_index):
        super(VirtualizeWorkerThread, self).__init__()
        self.images = images
        self.img_save_dir = img_save_dir
        self.index = index
        self.img_name_pre = img_name_pre
        self.is_fl = is_fl
        if is_fl:
            self.img_name_pre += "_fl"
        self.vir_index = vir_index
        self._isRunning = False

    def __del__(self):
        self.wait()

    def run(self):
        self._isRunning = True
        try:
            img_saved_path = build_and_draw_img_of_features(self.images, self.index, img_name_pre=self.img_name_pre,
                                                            img_save_dir=self.img_save_dir)
            if self._isRunning:
                print("VirtualizeWorkerThread Over: index: %d conv_index: %d, is_fl: %d" % (self.vir_index, self.index+1, self.is_fl))
                self.signal.emit(img_saved_path, self.index, self.is_fl, self.vir_index)
        except:
            traceback.print_exc()

    def stop(self):
        self._isRunning = False

    def isRunning(self):
        return self._isRunning


# 可视化模型输出的各类概率的工作线程
class VirtualizeBarDistributeThread(QThread):
    #  通过类成员对象定义信号对象
    signal = pyqtSignal(str, bool, int)

    def __init__(self, output_map, softmax_rate, img_name_pre, img_save_dir, is_fl, vir_index):
        super(VirtualizeBarDistributeThread, self).__init__()
        self.output_map = output_map
        self.softmax_rate = softmax_rate
        self.img_save_dir = img_save_dir
        self.img_name_pre = img_name_pre
        self.is_fl = is_fl
        if is_fl:
            self.img_name_pre += "_fl"
        self.vir_index = vir_index
        self._isRunning = False

    def __del__(self):
        self.wait()

    def run(self):
        self._isRunning = True
        try:
            if self.is_fl:
                bar_color = (118 / 255, 141 / 255, 50 / 255, 200 / 255)
            else:
                bar_color = (40 / 255, 135 / 255, 114 / 255, 200 / 255)
            img_saved_path = build_and_draw_bar_img(self.output_map, self.softmax_rate, img_name_pre=self.img_name_pre,
                                                    img_save_dir=self.img_save_dir, bar_color=bar_color)
            if self._isRunning:
                print("VirtualizeBarDistributeThread Over: index: %d, is_fl: %d" % (self.vir_index, self.is_fl))
                self.signal.emit(img_saved_path, self.is_fl, self.vir_index)
        except:
            traceback.print_exc()

    def stop(self):
        self._isRunning = False

    def isRunning(self):
        return self._isRunning
