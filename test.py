# coding=utf-8
'''
本文件是该项目的test文件，定义了各个训练参数，以及不同数据集，不同网络的选择；
'''
import os
import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import argparse
import time

from networks.ACNN import ACNN
from networks.ACCNN import ACCNN
from networks.AlexNet import AlexNet
from networks.VGG import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from networks.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from dal.JAFFE_DataSet import JAFFE
from dal.CKPlus_DataSet import CKPlus
from dal.FER2013_DataSet import FER2013
from dal.Data_Prefetcher import DataPrefetcher as Prefetcher
from dal.Raw_DataSet import RawDataSet
import transforms.transforms as transforms
import utils.utils as utils

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
print('cuda available: ', use_cuda)
print('using DEVICE: ', DEVICE)
enabled_nets = ["ACNN", "ACCNN", "AlexNet", "VGG11", "VGG13", "VGG16", "VGG19", "ResNet18", "ResNet34", "ResNet50",
                "ResNet101", "ResNet152"]
enabled_datasets = ["JAFFE", "CK+", "FER2013", "RAF"]

parser = argparse.ArgumentParser(description='PyTorch CNN Training With JAFFE')
# 模型选择
# parser.add_argument('--model', type=str, default='ACNN', help='CNN architecture')
parser.add_argument('--model', type=str, default='ACCNN', help='CNN architecture')
# parser.add_argument('--model', default='AlexNet', type=str, help='CNN architecture')
# parser.add_argument('--model', default='VGG11', type=str, help='CNN architecture')
# parser.add_argument('--model', default='VGG13', type=str, help='CNN architecture')
# parser.add_argument('--model', default='VGG16', type=str, help='CNN architecture')
# parser.add_argument('--model', default='VGG19', type=str, help='CNN architecture')
# parser.add_argument('--model', default='ResNet18', type=str, help='CNN architecture')
# parser.add_argument('--model', default='ResNet34', type=str, help='CNN architecture')
# parser.add_argument('--model', default='ResNet50', type=str, help='CNN architecture')
# parser.add_argument('--model', default='ResNet101', type=str, help='CNN architecture')
# parser.add_argument('--model', default='ResNet152', type=str, help='CNN architecture')

# 数据集选择，以及对于数据预处理，是否使用TenCrop进行数据增强
parser.add_argument('--dataset', default='JAFFE', type=str, help='dataset')
parser.add_argument('--tr_using_crop', default=True, type=bool, help='whether using TenCrop in data transform')
# parser.add_argument('--dataset', default='CK+', type=str, help='dataset')
# parser.add_argument('--tr_using_crop', default=False, type=bool, help='whether using TenCrop in data transform')
# parser.add_argument('--dataset', default='FER2013', type=str, help='dataset')
# parser.add_argument('--tr_using_crop', default=False, type=bool, help='whether using TenCrop in data transform')

# Other Parameters
# 是否使用面部标记点进行训练
parser.add_argument('--fl', default=True, type=bool, help='whether to use face landmarks to train')
# 存储的模型序号
parser.add_argument('--save_number', default=5, type=int, help='save_number')
# 批次大小
parser.add_argument('--bs', default=32, type=int, help='batch_size')
# 文件路径
parser.add_argument('--saved_model_name', default='Best_model.t7', type=str, help='saved_model_name')

opt = parser.parse_args()

test_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
Test_acc = 0.

print("------------Preparing Model...----------------")
n_classes = 7
net_to_save_dir = "Saved_Models"
net_to_save_path = os.path.join(net_to_save_dir, str(opt.save_number),
                                opt.dataset + '_' + opt.model + '_' + str(opt.save_number))

if opt.model.lower() == "ACNN".lower():
    net = ACNN(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ACCNN".lower():
    net = ACCNN(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "AlexNet".lower():
    net = AlexNet(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "VGG11".lower():
    net = vgg11_bn(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "VGG13".lower():
    net = vgg13_bn(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "VGG16".lower():
    net = vgg16_bn(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "VGG19".lower():
    net = vgg19_bn(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ResNet18".lower():
    net = resnet18(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ResNet34".lower():
    net = resnet34(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ResNet50".lower():
    net = resnet50(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ResNet101".lower():
    net = resnet101(n_classes=n_classes).to(DEVICE)
elif opt.model.lower() == "ResNet152".lower():
    net = resnet152(n_classes=n_classes).to(DEVICE)
else:
    net = None
    assert ("opt.model should be in %s, but got %s" % (enabled_nets, opt.model))
# Load checkpoint.
print('==> Loading Model Parameters...')
assert os.path.isdir(net_to_save_path), 'Error: no checkpoint directory found in path: %s' % (net_to_save_path)
assert os.path.exists(os.path.join(net_to_save_path, opt.saved_model_name)), 'Error: no checkpoint file found in path: %s' % (os.path.join(net_to_save_path, opt.saved_model_name))
checkpoint = torch.load(os.path.join(net_to_save_path, opt.saved_model_name))
net.load_state_dict(checkpoint['net'])
test_acc_map['best_acc'] = checkpoint['best_test_acc']
test_acc_map['best_acc_epoch'] = checkpoint['best_test_acc_epoch']
start_epoch = checkpoint['cur_epoch'] + 1
print('==> Loading Model Parameters Over!')

# for gray images
IMG_MEAN = [0.449]
IMG_STD = [0.226]
# for RGB images
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]

crop_img_size = int(net.input_size * 1.2)
input_img_size = net.input_size
transform_using_crop = opt.tr_using_crop
if transform_using_crop:
    transform_test = transforms.Compose([
        transforms.Resize(crop_img_size),
        transforms.TenCrop(input_img_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(IMG_MEAN, IMG_STD)(
            transforms.ToTensor()(crop)) for crop in crops])),
    ])
else:
    transform_test = transforms.Compose([
        transforms.Resize(input_img_size),  # 缩放将图片的最小边缩放为 input_img_size，因此如果输入是非正方形的，那么输出也不是正方形的
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

# 损失函数
criterion, target_type = nn.CrossEntropyLoss(), 'ls'

print("------------Preparing Data...----------------")
if opt.dataset == "JAFFE":
    test_data = JAFFE(is_train=False, transform=transform_test, target_type=target_type, using_fl=opt.fl)
# elif opt.dataset == "CK+48":
#     test_data = CKPlus(is_train=False, transform=transform_test, target_type=target_type, img_dir_pre_path="data/CK+48")
elif opt.dataset == "CK+":
    test_data = CKPlus(is_train=False, transform=transform_test, target_type=target_type, using_fl=opt.fl)
elif opt.dataset == "FER2013":
    test_data = FER2013(is_train=False, private_test=True, transform=transform_test, target_type=target_type,
                        using_fl=opt.fl)
else:
    assert ("opt.dataset should be in %s, but got %s" % (enabled_datasets, opt.dataset))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.bs, shuffle=False)
test_prefetcher = None
print("------------%s Data Already be Prepared------------" % opt.dataset)

# Testing
def old_test():
    '''未使用DataPrefetcher'''
    global Test_acc
    private_test_loss = 0
    net.eval()
    correct = 0
    total = 0
    cur_test_acc = 0.
    correct_map = [0, 0, 0, 0, 0, 0, 0]
    time_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs, targets = put_through_net(inputs, targets)

            loss = criterion(outputs, targets)
            private_test_loss += float(loss.data)
            _, predicted = torch.max(outputs.data, 1)
            if target_type == 'ls':
                ground_value = targets.data
            elif target_type == 'fa':
                _, ground_value = torch.max(targets.data, 1)

            for i in range(len(predicted)):
                if predicted[i] == ground_value[i]:
                    c = predicted[i].item()
                    test_acc_map[c] += 1
                    correct_map[c] += 1

            total += targets.size(0)
            correct += predicted.eq(ground_value.data).cpu().sum()
            cur_test_acc = float(correct) / float(total) * 100.

            time_end = time.time()
            duration = time_end - time_start
            utils.progress_bar(batch_idx, len(test_loader), 'Time: %.2fs | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                               (duration, private_test_loss / (batch_idx + 1), cur_test_acc, correct, total))

            # 删除无用的变量，释放显存
            del loss
            del inputs
            del outputs
            del predicted

    Test_acc = cur_test_acc

# Testing
def test():
    '''使用DataPrefetcher加速'''
    print("---Test---")
    global Test_acc
    private_test_loss = 0
    net.eval()
    correct = 0
    total = 0
    cur_test_acc = 0.
    correct_map = [0, 0, 0, 0, 0, 0, 0]
    time_start = time.time()
    with torch.no_grad():
        batch_idx = 0
        inputs, targets = test_prefetcher.next()
        while inputs is not None:
            outputs, targets = put_through_net(inputs, targets)

            loss = criterion(outputs, targets)
            private_test_loss += float(loss.data)
            _, predicted = torch.max(outputs.data, 1)
            if target_type == 'ls':
                ground_value = targets.data
            elif target_type == 'fa':
                _, ground_value = torch.max(targets.data, 1)

            for i in range(len(predicted)):
                if predicted[i] == ground_value[i]:
                    c = predicted[i].item()
                    test_acc_map[c] += 1
                    correct_map[c] += 1

            total += targets.size(0)
            correct += predicted.eq(ground_value.data).cpu().sum()
            cur_test_acc = float(correct) / float(total) * 100.

            time_end = time.time()
            duration = time_end - time_start
            utils.progress_bar(batch_idx, len(test_loader), 'Time: %.2fs | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                               (duration, private_test_loss / (batch_idx + 1), cur_test_acc, correct, total))

            # 删除无用的变量，释放显存
            del loss
            del inputs
            del outputs
            del predicted

            inputs, targets = test_prefetcher.next()
            batch_idx += 1

    Test_acc = cur_test_acc


def put_through_net(inputs, targets):
    '''
    将inputs输入net，得到输出，并返回
    :param inputs: 网络的输入
    :param targets: 网络输入对应的label
    :return: 网络的输出
    '''
    if transform_using_crop:
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        targets = torch.Tensor([[target]*ncrops for target in targets]).view(-1)
    if use_cuda:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE, torch.long)
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    return outputs, targets


if __name__ == "__main__":
    if not os.path.isdir(net_to_save_dir):
        os.mkdir(net_to_save_dir)
    if not os.path.isdir(os.path.join(net_to_save_dir, str(opt.save_number))):
        os.mkdir(os.path.join(net_to_save_dir, str(opt.save_number)))
    if not os.path.isdir(net_to_save_path):
        os.mkdir(net_to_save_path)
    if use_cuda:
        test_prefetcher = Prefetcher(test_loader)
        test()
    else:
        old_test()
    print(test_acc_map)
