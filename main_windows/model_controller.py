# coding=utf-8
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
sys.path.append("..")
from networks.ACCNN import ACCNN
import transforms.transforms as transforms
from utils.face_recognition import crop_face_area_and_get_landmarks, get_img_with_landmarks

IMG_MEAN = [0.449]
IMG_STD = [0.226]
use_cuda = torch.cuda.is_available()
use_cuda = False
DEVICE = torch.device("cuda" if use_cuda else "cpu")


class ModelController():
    '''
    用于系统对于模型的控制，以及使用模型进行表情识别
    '''
    def __init__(self, model_root_pre_path='', dataset='FER2013', tr_using_crop=False, *args, **kwargs):
        self.model_root_pre_path = model_root_pre_path

        self.model = ACCNN(7, pre_trained=True, dataset=dataset, root_pre_path=model_root_pre_path, fold=5, virtualize=True,
                           using_fl=False).to(DEVICE)
        self.model_fl = ACCNN(7, pre_trained=True, dataset=dataset, root_pre_path=model_root_pre_path, fold=5, virtualize=True,
                              using_fl=True).to(DEVICE)
        self.tr_using_crop = tr_using_crop
        if self.tr_using_crop:
            crop_img_size = int(self.model.input_size * 1.2)
            self.transform_test = transforms.Compose([
                transforms.Resize(crop_img_size),
                transforms.TenCrop(self.model.input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(IMG_MEAN, IMG_STD)(transforms.ToTensor()(crop)) for crop in crops])),
            ])
        else:
            self.transform_test = transforms.Compose([
                transforms.Resize(int(self.model.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_STD),
            ])

    def fer_recognization(self, img_arr, weights_fl=0.4):
        """
        人脸识别
        :param img_arr: img的numpy array对象
        :param weights_fl: 对于face_landmarks的权重，0~1之间
        :param resize_size:
        :return:
        """
        # 模型的输入准备
        start_time = time.time()
        img = Image.fromarray(img_arr).convert("L")
        img, face_box, face_landmarks = crop_face_area_and_get_landmarks(img)
        landmarks_img = get_img_with_landmarks(img, face_landmarks)
        inputs = self.transform_test(img)
        inputs_fl = self.transform_test(landmarks_img)
        self.clean_model_features_out()
        pre_exec_data_end_time = time.time()
        pre_exec_data_duration = round((pre_exec_data_end_time - start_time) * 1000, 2)
        # 模型输入
        outputs, vir = self.model_test(self.model, inputs)
        outputs_fl, vir_fl = self.model_test(self.model_fl, inputs_fl)

        # 对输出进行处理
        real_outputs = (1 - weights_fl) * outputs.data + weights_fl * outputs_fl.data
        _, predicted = torch.max(real_outputs, 1)  # 此处 1 表示维度
        predict_end_time = time.time()
        predict_duration = round((predict_end_time - pre_exec_data_end_time) * 1000, 2)
        # print(predicted)
        softmax_rate = (
            np.array(outputs.cpu().data[0]), np.array(outputs_fl.cpu().data[0]), np.array(real_outputs.cpu().data[0]))
        return (face_box, self.model.output_map[predicted.item()], softmax_rate, [vir, vir_fl], [pre_exec_data_duration,
                                                                                                 predict_duration])

    def model_test(self, model, inputs):
        bs = 1
        ncrops = 1
        if self.tr_using_crop:
            ncrops, c, h, w = np.shape(inputs)
        else:
            c, h, w = np.shape(inputs)
        model.clean_features_out()

        # 模型中间层可视化抽取准备
        features_hook = []

        def get_features_hook(self, input, output):
            features_hook.append(input)

        handlers = []
        for layer in model.named_modules():
            if isinstance(layer[1], nn.Conv2d):
                handlers.append(layer[1].register_forward_hook(get_features_hook))

        # 模型测试和识别
        model.eval()
        with torch.no_grad():
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs = inputs.to(DEVICE)
            inputs = Variable(inputs)
            outputs = model(inputs)
            if self.tr_using_crop:
                outputs = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            else:
                outputs = outputs

        # 删除绑定
        for handler in handlers:
            handler.remove()
        # 读取抽取的中间层输出
        virtualizations = []
        for conv_number in range(len(features_hook)):
            virtualizations.append(features_hook[conv_number][0].cpu())
        virtualizations.append(model.features_out[0].cpu())
        return outputs, virtualizations

    def clean_model_features_out(self):
        """
        清空特征层的输出缓存
        :return: 无
        """
        self.model.clean_features_out()
        self.model_fl.clean_features_out()

