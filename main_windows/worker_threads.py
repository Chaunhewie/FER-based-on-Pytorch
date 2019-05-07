# coding=utf-8
import time
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from main_windows.model_controller import ModelController
import traceback

# 初始化模型和模型参数的工作线程
class InitModelThread(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(ModelController)

    def __init__(self, model_root_pre_path, dataset, tr_using_crop):
        super(InitModelThread, self).__init__()
        self.model_root_pre_path = model_root_pre_path
        self.dataset = dataset
        self.tr_using_crop = tr_using_crop

    def __del__(self):
        self.wait()

    def run(self):
        model_controller = ModelController(model_root_pre_path=self.model_root_pre_path, dataset=self.dataset, tr_using_crop=self.tr_using_crop)
        self._signal.emit(model_controller)  # 注意这里与_signal = pyqtSignal(str)中的类型相同


# 人脸识别的工作线程
class FERWorkerThread(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(QPixmap, tuple, float)

    def __init__(self, model_controller, img_origin):
        super(FERWorkerThread, self).__init__()
        self.model_controller = model_controller
        self.img_origin = img_origin
        self.QPixmap_Channels_Count = 4

    def __del__(self):
        self.wait()

    def run(self):
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
            self._signal.emit(self.img_origin, res, duration)
        except:
            traceback.print_exc()
