# coding=utf-8
import os
import torch
from PIL import Image
from math import ceil
import torch.nn as nn
import numpy as np
import utils


def draw_features_of_net(net, test_inputs, img_name_pre="", blank_size=2, img_save_dir="Saved_Virtualizations"):
    """
    绘制网络所有的特征层的输出情况
    :param net: 网络
    :param test_inputs: 测试的输入
    :param img_name_pre: 存储的图片的前缀
    :param blank_size: 样图之间的间距（pixel）
    :param img_save_dir: 存储的目录
    :return: None
    """
    # 绑定并测试
    features_hook = []

    def get_features_hook(self, input, output):
        features_hook.append(input)

    handlers = []
    for layer in net.named_modules():
        if isinstance(layer[1],nn.Conv2d):
            handlers.append(layer[1].register_forward_hook(get_features_hook))
    with torch.no_grad():
        net(test_inputs)
        features_hook.append(net.features_out)
    for handler in handlers:
        handler.remove()

    # 可视化展示
    for layer_number in range(len(features_hook)):
        images = np.array(features_hook[layer_number][0].cpu())
        # print(images.shape)
        img_save_name = img_name_pre + "_feature_layer_" + str(layer_number)
        for i in range(len(images)):
            # print("-----------img %s------------" % str(i))
            image = images[i]
            # print("image shape:", image.shape)
            image_shape = image.shape
            print(image_shape)
            col_num = int(ceil(image_shape[0] ** 0.5))
            row_num = int(ceil(image_shape[0] / col_num))
            height = row_num * image_shape[1]
            width = col_num * image_shape[2]
            # print(col_num, row_num, height, width)
            img_arr = np.array([[0.5 for _ in range(width+(col_num-1)*blank_size)]for _ in range(height+(row_num-1)*blank_size)])
            for j in range(len(image)):
                start_row_index = (j//col_num)*(blank_size+image_shape[1])
                start_col_index = (j % col_num)*(blank_size+image_shape[2])
                for row_pixel_index in range(image_shape[1]):
                    for col_pixel_index in range(image_shape[2]):
                        img_arr[start_row_index+row_pixel_index][start_col_index+col_pixel_index] = \
                            image[j][row_pixel_index][col_pixel_index]
            img = Image.fromarray(img_arr)
            if not os.path.exists(img_save_dir):
                os.mkdir(img_save_dir)
            utils.draw_img(img, os.path.join(img_save_dir, img_save_name+"_of_img_"+str(i)), plt_show=False)
            break

def test_draw_features_and_weights_of_net():
    import sys
    sys.path.append("..")
    from torch.autograd import Variable
    import transforms.transforms as transforms
    from networks.ACNN import ACNN
    from networks.ACCNN import ACCNN
    from networks.AlexNet import AlexNet
    from dal.CKPlus48_DataSet import CKPlus48
    from dal.FER2013_DataSet import FER2013
    from dal.JAFFE_DataSet import JAFFE

    net_to_save_dir = "../Saved_Models"
    saved_model_name = 'Best_model.t7'
    fold = 2
    enabled_nets = ["ACNN", "AlexNet"]
    enabled_datasets = ["JAFFE", "CK+48", "CK+", "FER2013"]

    # 配置信息
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    net_name, n_classes = 'ACNN', 7

    if net_name == "ACNN":
        net = ACNN(n_classes=n_classes).to(DEVICE)
    elif net_name == "AlexNet":
        net = AlexNet(n_classes=n_classes).to(DEVICE)
    input_img_size = net.input_size
    transform_train = transforms.Compose([
        transforms.Resize(input_img_size),  # 缩放将图片的最小边缩放为 input_img_size，因此如果输入是费正方形的，那么输出也不是正方形的
        transforms.RandomCrop(input_img_size),  # 用于将非正方形的图片进行处理
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_img_size = int(input_img_size * 1.1)  # 测试时，图片resize大小
    transform_test = transforms.Compose([
        transforms.Resize(test_img_size),
        transforms.TenCrop(input_img_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    dataset = 'CK+'
    target_type = 'ls'
    if dataset == "JAFFE":
        test_data = JAFFE(is_train=False, transform=transform_test, target_type=target_type,
                          img_dir_pre_path="../data/jaffe")
    elif dataset == "CK+48":
        test_data = CKPlus48(is_train=False, transform=transform_test, target_type=target_type,
                             img_dir_pre_path="../data/CK+48")
    elif dataset == "CK+":
        test_data = CKPlus48(is_train=False, transform=transform_test, target_type=target_type,
                             img_dir_pre_path="../data/CK+")
    elif dataset == "FER2013":
        test_data = FER2013(is_train=False, private_test=True, transform=transform_test, target_type=target_type,
                            img_dir_pre_path="../data/fer2013")
    else:
        assert ("opt.dataset should be in %s, but got %s" % (enabled_datasets, dataset))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    # 获取model
    net_to_save_path = os.path.join(net_to_save_dir, dataset + '_' + net_name + "_" + str(fold))
    print("get net parameters from:", net_to_save_path)
    checkpoint = torch.load(os.path.join(net_to_save_path, saved_model_name))
    net.load_state_dict(checkpoint['net'])
    print("---------------net: %s, dataset: %s---------------" % (net_name, dataset))

    # 获取数据
    inputs, targets = next(iter(test_loader))
    # print(inputs.shape, targets)
    # input = inputs[0][0][0]
    # print(input.shape)
    # draw_img(input)

    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    if use_cuda:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    inputs, targets = Variable(inputs), Variable(targets)

    draw_features_of_net(net, inputs, img_name_pre='test', img_save_dir="../Saved_Virtualizations")
    draw_weights_of_net(net, img_name_pre='test', img_save_dir="../Saved_Virtualizations")


def draw_weights_of_net(net, img_name_pre = "", blank_size = 2, img_save_dir = "Saved_Virtualizations"):
    """
    绘制网络所有的特征层的输出情况
    :param net: 网络
    :param img_name_pre: 存储的图片的前缀
    :param blank_size: 样图之间的间距（pixel）
    :param img_save_dir: 存储的目录
    :return: None
    """
    layer_number = 0
    stat_dict = net.state_dict()
    for dic in stat_dict.keys():
        dic_splited = dic.split(".")
        if dic_splited[0] == 'features' and dic_splited[2] == 'weight':
            img_save_name = img_name_pre + "_weight_layer_" + str(layer_number)
            weights = stat_dict[dic]
            # print(dic, weights)
            out_channel_num = len(weights)
            in_channel_num = len(weights[0])
            kernel_height = len(weights[0][0])
            kernel_width = len(weights[0][0][0])
            img_arr = np.array([[0.5 for _ in range(kernel_width*in_channel_num+(in_channel_num-1)*blank_size)]for _ in range(kernel_height*out_channel_num+(out_channel_num-1)*blank_size)])
            # print(img_arr.shape)
            for out_channel_number in range(out_channel_num):
                for in_channel_number in range(in_channel_num):
                    start_row_index, start_col_index = (out_channel_number)*(blank_size+kernel_height), (in_channel_number)*(blank_size+kernel_width)
                    kernel = weights[out_channel_number][in_channel_number]
                    # print(kernel.shape)
                    for row_pixel_index in range(kernel.shape[0]):
                        for col_pixel_index in range(kernel.shape[1]):
                            img_arr[start_row_index+row_pixel_index][start_col_index+col_pixel_index] = kernel[row_pixel_index][col_pixel_index]
            img = Image.fromarray(img_arr)
            if not os.path.exists(img_save_dir):
                os.mkdir(img_save_dir)
            utils.draw_img(img, os.path.join(img_save_dir, img_save_name), plt_show=False)
            layer_number += 1


if __name__ == "__main__":
    test_draw_features_and_weights_of_net()
