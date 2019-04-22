# coding=utf-8
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import sys
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
sys.path.append("..")
from networks.ACCNN import ACCNN
import transforms.transforms as transforms

class FERWindow(QMainWindow):

    def __init__(self, root_pre_path='main_windows', *args, **kwargs):
        super(FERWindow, self).__init__(*args, **kwargs)

        self.resize(800, 500)

        self.show_pic_label = QLabel(self)
        self.show_pic_label.setText("   显示图片")
        self.show_pic_label.setFixedSize(350, 350)
        self.show_pic_label.move(20, 90)

        self.show_pic_label.setStyleSheet("QLabel{background:white;}QLabel{color:rgb(300,300,300,120);"
                                          "font-size:10px;font-weight:bold;font-family:宋体;}")

        self.show_res_label = QLabel(self)
        self.show_res_label.setText("   显示图片")
        self.show_res_label.setFixedSize(350, 350)
        self.show_res_label.move(430, 90)

        self.show_res_label.setStyleSheet("QLabel{background:white;}QLabel{color:rgb(300,300,300,120);"
                                          "font-size:10px;font-weight:bold;font-family:宋体;}")

        self.get_pic_btn = QPushButton(self)
        self.get_pic_btn.setText("打开图片")
        self.get_pic_btn.move(10, 30)
        self.get_pic_btn.clicked.connect(self.open_image)

        self.setWindowTitle("My FER Program")
        self.show()

        self.use_cuda = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.use_cuda else "cpu")
        if len(root_pre_path) <= 0:
            self.model_init('..')
        else:
            self.model_init()

    def model_init(self, model_root_pre_path=''):
        self.model = ACCNN(7, True, model_root_pre_path).to(self.DEVICE)
        self.transform_test = transforms.Compose([
            transforms.Resize(int(self.model.input_size*1.1)),
            transforms.TenCrop(self.model.input_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])


    def open_image(self):
        # 打开并展示一张图片
        img_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;*.jpg;;All Files(*)")
        print("get image path: ", img_path)
        img_origin = QPixmap(img_path)
        if img_origin.width() > img_origin.height():
            img_origin = img_origin.scaledToWidth(self.show_pic_label.width())
        else:
            img_origin = img_origin.scaledToHeight(self.show_pic_label.height())
        self.show_pic_label.setPixmap(img_origin)


        # 输入打开的图片到模型中识别并将结果展示
        img_res = self.fer_recognization(img_path)
        # print(img_res)
        self.show_res_label.setText(img_res)
        # if img_res.width() > img_res.height():
        #     img_res = img_res.scaledToWidth(self.show_res_label.width())
        # else:
        #     img_res = img_res.scaledToHeight(self.show_res_label.height())
        # self.show_res_label.setPixmap(img_res)


    def fer_recognization(self, img_path):
        img = Image.open(img_path)
        img = img.convert("L")
        inputs = self.transform_test(img)
        bs = 1
        ncrops, c, h, w = np.shape(inputs)
        # print(np.shape(inputs))
        with torch.no_grad():
            inputs = inputs.view(-1, c, h, w)
            if self.use_cuda:
                inputs = inputs.to(self.DEVICE)
            inputs = Variable(inputs)
            outputs = self.model(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            _, predicted = torch.max(outputs_avg.data, 1)
            # print(predicted)
        return self.model.output_map[predicted.item()]


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")

    window = FERWindow(root_pre_path='')
    app.exec_()