﻿# coding=utf-8
'''
本文件是该项目的main文件，定义了各个训练参数，以及不同数据集，不同网络的选择；
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
# parser.add_argument('--dataset', default='JAFFE', type=str, help='dataset')
# parser.add_argument('--tr_using_crop', default=True, type=bool, help='whether using TenCrop in data transform')
# parser.add_argument('--dataset', default='CK+', type=str, help='dataset')
# parser.add_argument('--tr_using_crop', default=False, type=bool, help='whether using TenCrop in data transform')
parser.add_argument('--dataset', default='FER2013', type=str, help='dataset')
parser.add_argument('--tr_using_crop', default=False, type=bool, help='whether using TenCrop in data transform')

# Other Parameters
# 是否使用面部标记点进行训练
parser.add_argument('--fl', default=True, type=bool, help='whether to use face landmarks to train')
# 存储的模型序号
parser.add_argument('--save_number', default=5, type=int, help='save_number')
# 批次大小
parser.add_argument('--bs', default=32, type=int, help='batch_size')
# 学习率
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# epoch
parser.add_argument('--epoch', default=200, type=int, help='training epoch num')
# 每次获得到更优的准确率后，会进行一次存储，此选项选择是否从上次存储位置继续
parser.add_argument('--resume', default=True, type=bool, help='resume training from last checkpoint')
# 表示默认一开始的前 $lre_je 次epoch，增大lr来进行跳跃，解决一开始收敛缓慢问题
parser.add_argument('--lre_je', default=20, type=int, help='learning rate expand jump epoch')
# 表示默认从第 $lrd_se 次epoch开始进行lr的递减
parser.add_argument('--lrd_se', default=180, type=int, help='learning rate decay start epoch')
# 表示默认每经过2次epoch进行一次递减
parser.add_argument('--lrd_s', default=2, type=int, help='learning rate decay step')
# 表示每次的lr的递减率，默认每递减一次乘一次0.9
parser.add_argument('--lrd_r', default=0.9, type=float, help='learning rate decay rate')
# 预测错误数据集进行训练的最大迭代次数，0为不迭代
parser.add_argument('--pred_err_epoch_max', default=5, type=int, help='max epoch of predicted err dataset')
# 预测错误数据集进行训练的学习率为正常训练学习率的递减比例： new_lr = lr / pred_err_lr_decay
parser.add_argument('--pred_err_lr_decay', default=1, type=float, help='lr decays when in epoch of predicted err dataset')

opt = parser.parse_args()

train_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
test_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
Train_acc, Test_acc = 0., 0.

print("------------Preparing Model...----------------")
n_classes = 7
net_to_save_dir = "Saved_Models"
net_to_save_path = os.path.join(net_to_save_dir, str(opt.save_number),
                                opt.dataset + '_' + opt.model + '_' + str(opt.save_number))
if opt.fl:
    saved_model_name = "Best_model_fl.t7"
    saved_temp_model_name = "Best_model_fl_temp.t7"
    model_over_flag_name = "__%d_success_fl__" % (opt.epoch)
    history_file_name = "history_fl.txt"
else:
    saved_model_name = "Best_model.t7"
    saved_temp_model_name = "Best_model_temp.t7"
    model_over_flag_name = "__%d_success__" % (opt.epoch)
    history_file_name = "history.txt"
over_flag = False  # 如果已经成功训练完，就可以结束了
TEMP_EPOCH = 2  # 用于暂时存储，每TEMP_EPOCH次存一次
temp_internal = TEMP_EPOCH
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
start_epoch = 0
if opt.resume:
    # Load checkpoint.
    print('==> Loading Model Parameters...')
    if os.path.exists(os.path.join(net_to_save_path, saved_temp_model_name)):
        if os.path.exists(os.path.join(net_to_save_path, model_over_flag_name)):
            print("Model trained over flag checked!")
            over_flag = True
        assert os.path.isdir(net_to_save_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(net_to_save_path, saved_temp_model_name))
        net.load_state_dict(checkpoint['net'])
        test_acc_map['best_acc'] = checkpoint['best_test_acc']
        test_acc_map['best_acc_epoch'] = checkpoint['best_test_acc_epoch']
        start_epoch = checkpoint['cur_epoch'] + 1
    else:
        print("Checkout File not Found, No initialization.")
print("------------%s Model Already be Prepared------------" % opt.model)

if not over_flag:
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
        transform_train = transforms.Compose([
            transforms.Resize(crop_img_size),
            transforms.TenCrop(input_img_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(IMG_MEAN, IMG_STD)(
                transforms.ToTensor()(
                    transforms.RandomHorizontalFlip()(
                        transforms.RandomRotation(30)(crop)))) for crop in crops])),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(crop_img_size),
            transforms.TenCrop(input_img_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(IMG_MEAN, IMG_STD)(
                transforms.ToTensor()(crop)) for crop in crops])),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(input_img_size),  # 缩放将图片的最小边缩放为 input_img_size，因此如果输入是非正方形的，那么输出也不是正方形的
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_img_size),  # 缩放将图片的最小边缩放为 input_img_size，因此如果输入是非正方形的，那么输出也不是正方形的
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

    # criterion, target_type = nn.MSELoss(), 'fa'
    criterion, target_type = nn.CrossEntropyLoss(), 'ls'
    # 随机梯度下降 优化
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    # Adam 优化
    # optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=5e-4)
    # Adadelta 优化
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=opt.lr, weight_decay=5e-4)

    print("------------Preparing Data...----------------")
    if opt.dataset == "JAFFE":
        train_data = JAFFE(is_train=True, transform=transform_train, target_type=target_type, using_fl=opt.fl)
        test_data = JAFFE(is_train=False, transform=transform_test, target_type=target_type, using_fl=opt.fl)
    # elif opt.dataset == "CK+48":
    #     train_data = CKPlus(is_train=True, transform=transform_train, target_type=target_type, img_dir_pre_path="data/CK+48")
    #     test_data = CKPlus(is_train=False, transform=transform_test, target_type=target_type, img_dir_pre_path="data/CK+48")
    elif opt.dataset == "CK+":
        train_data = CKPlus(is_train=True, transform=transform_train, target_type=target_type, using_fl=opt.fl)
        test_data = CKPlus(is_train=False, transform=transform_test, target_type=target_type, using_fl=opt.fl)
    elif opt.dataset == "FER2013":
        train_data = FER2013(is_train=True, private_test=True, transform=transform_train, target_type=target_type,
                             using_fl=opt.fl)
        test_data = FER2013(is_train=False, private_test=True, transform=transform_test, target_type=target_type,
                            using_fl=opt.fl)
    else:
        assert ("opt.dataset should be in %s, but got %s" % (enabled_datasets, opt.dataset))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.bs, shuffle=False)
    train_prefetcher = None
    test_prefetcher = None
    print("------------%s Data Already be Prepared------------" % opt.dataset)


# Training
def old_train(epoch):
    '''未使用DataPrefetcher'''
    # 根据训练的epoch次数来降低learning rate
    if epoch >= opt.lrd_se > 0:
        frac = ((epoch - opt.lrd_se) // opt.lrd_s) + 1
        decay_factor = opt.lrd_r ** frac
        current_lr = opt.lr * decay_factor  # current_lr = opt.lr * 降低率 ^ ((epoch - 开始decay的epoch) // 每次decay的epoch num)
        utils.set_lr(optimizer, current_lr)  # set the learning rate
    else:
        current_lr = opt.lr
    if epoch < opt.lre_je:
        current_lr *= 1.5  # 解决一开始收敛慢的问题
    print('learning_rate: %s' % str(current_lr))
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    cur_train_acc = 0.
    time_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs, targets = put_through_net(inputs, targets)
        # print("outputs:", outputs)
        # print("targets:", targets)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 2*current_lr)  # 解决梯度爆炸 https://blog.csdn.net/u010814042/article/details/76154391
        optimizer.step()

        train_loss += float(loss.data)
        _, predicted = torch.max(outputs.data, 1)  # torch.max() 加上dim参数后，返回值为 max_value, max_value_index
        if target_type == 'ls':
            ground_value = targets.data
        elif target_type == 'fa':
            _, ground_value = torch.max(targets.data, 1)
        # print("predicted:", predicted)
        # print("ground_value:", ground_value)

        for i in range(len(predicted)):
            if predicted[i] == ground_value[i]:
                train_acc_map[predicted[i].item()] += 1

        total += targets.size(0)
        correct += predicted.eq(ground_value.data).cpu().sum()
        # print("equal: ", predicted.eq(ground_value.data).cpu())
        cur_train_acc = float(correct) / float(total) * 100.

        time_end = time.time()
        duration = time_end - time_start
        utils.progress_bar(batch_idx, len(train_loader), 'Time: %.2fs | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                           (duration, train_loss / (batch_idx + 1), cur_train_acc, correct, total))

        # 删除无用的变量，释放显存
        del loss
        del inputs
        del outputs
        del predicted
    Train_acc = cur_train_acc
    write_history('Train', epoch, cur_train_acc, train_loss / (batch_idx + 1), None)


# Testing
def old_test(epoch):
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
    if test_acc_map['best_acc'] < Test_acc or (test_acc_map['best_acc']
                                               <= Test_acc and train_acc_map['best_acc'] <= Train_acc):
        train_acc_map['best_acc'] = Train_acc
        train_acc_map['best_acc_epoch'] = epoch
        test_acc_map['best_acc'] = Test_acc
        test_acc_map['best_acc_epoch'] = epoch
        print('Saving net to %s' % net_to_save_path)
        print('best_acc: %0.3f' % test_acc_map['best_acc'])
        print('correct_map: %s' % correct_map)
        state = {'net': net.state_dict() if use_cuda else net,
                 'best_test_acc': test_acc_map['best_acc'],
                 'best_test_acc_epoch': test_acc_map['best_acc_epoch'],
                 'best_train_acc': train_acc_map['best_acc'],
                 'best_train_acc_epoch': train_acc_map['best_acc_epoch'],
                 'cur_epoch': epoch,
                 'correct_map': correct_map,
                 }
        torch.save(state, os.path.join(net_to_save_path, saved_model_name))
    write_history('Test', epoch, cur_test_acc, private_test_loss / (batch_idx + 1), correct_map)


# Training
def train(epoch):
    '''使用DataPrefetcher加速'''
    print("---Train---")
    # 根据训练的epoch次数来降低learning rate
    if epoch >= opt.lrd_se > 0:
        frac = ((epoch - opt.lrd_se) // opt.lrd_s) + 1
        decay_factor = opt.lrd_r ** frac
        current_lr = opt.lr * decay_factor  # current_lr = opt.lr * 降低率 ^ ((epoch - 开始decay的epoch) // 每次decay的epoch num)
        utils.set_lr(optimizer, current_lr)  # set the learning rate
    else:
        current_lr = opt.lr
    if epoch < opt.lre_je:
        current_lr *= 1.5  # 解决一开始收敛慢的问题
    print('learning_rate: %s' % str(current_lr))
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pred_err_dataset = RawDataSet()
    pred_err_map = [0, 0, 0, 0, 0, 0, 0]
    cur_train_acc = 0.
    time_start = time.time()
    batch_idx = 0
    inputs, targets = train_prefetcher.next()
    while inputs is not None:
        optimizer.zero_grad()
        outputs, targets = put_through_net(inputs, targets)
        # print("outputs:", outputs)
        # print("targets:", targets)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 2*current_lr)  # 解决梯度爆炸 https://blog.csdn.net/u010814042/article/details/76154391
        optimizer.step()

        train_loss += float(loss.data)
        _, predicted = torch.max(outputs.data, 1)  # torch.max() 加上dim参数后，返回值为 max_value, max_value_index
        if target_type == 'ls':
            ground_value = targets.data
        elif target_type == 'fa':
            _, ground_value = torch.max(targets.data, 1)
        # print("predicted:", predicted)
        # print("ground_value:", ground_value)

        for i in range(len(predicted)):
            if predicted[i] == ground_value[i]:
                train_acc_map[predicted[i].item()] += 1
            else:
                pred_err_dataset.add(inputs[i], targets[i])
                pred_err_map[ground_value[i].item()] += 1

        total += targets.size(0)
        correct += predicted.eq(ground_value.data).cpu().sum()
        # print("equal: ", predicted.eq(ground_value.data).cpu())
        cur_train_acc = float(correct) / float(total) * 100.

        time_end = time.time()
        duration = time_end - time_start
        utils.progress_bar(batch_idx, len(train_loader), 'Time: %.2fs | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                           (duration, train_loss / (batch_idx + 1), cur_train_acc, correct, total))
        
        # 删除无用的变量，释放显存
        del loss
        del inputs
        del outputs
        del predicted
        
        inputs, targets = train_prefetcher.next()
        batch_idx += 1

    Train_acc = cur_train_acc
    write_history('Train', epoch, cur_train_acc, train_loss / (batch_idx + 1), None)

    pred_err_loop(current_lr/opt.pred_err_lr_decay, pred_err_dataset, pred_err_map)
    del pred_err_dataset


# Testing
def test(epoch):
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
    if test_acc_map['best_acc'] < Test_acc or (test_acc_map['best_acc'] <= Test_acc and train_acc_map['best_acc'] <= Train_acc):
        train_acc_map['best_acc'] = Train_acc
        train_acc_map['best_acc_epoch'] = epoch
        test_acc_map['best_acc'] = Test_acc
        test_acc_map['best_acc_epoch'] = epoch
        print('Saving net to %s' % net_to_save_path)
        print('best_acc: %0.3f' % test_acc_map['best_acc'])
        print('correct_map: %s' % correct_map)
        state = {'net': net.state_dict() if use_cuda else net,
                 'best_test_acc': test_acc_map['best_acc'],
                 'best_test_acc_epoch': test_acc_map['best_acc_epoch'],
                 'best_train_acc': train_acc_map['best_acc'],
                 'best_train_acc_epoch': train_acc_map['best_acc_epoch'],
                 'cur_epoch': epoch,
                 'correct_map': correct_map,
                 }
        torch.save(state, os.path.join(net_to_save_path, saved_model_name))
    write_history('Test', epoch, cur_test_acc, private_test_loss / (batch_idx + 1), correct_map)


def pred_err_loop(current_lr, pred_err_dataset, pred_err_map):
    '''
    loop循环，将预测错的继续训练
    :param current_lr: 学习率
    :param pred_err_dataset: 预测错误的数据集
    :param pred_err_map: 预测错误的类别数量
    :return: None
    '''
    pred_err_epoch = 1
    pred_err_epoch_max = opt.pred_err_epoch_max
    pred_err_dataset_temp = None
    while len(pred_err_dataset) > 0 and pred_err_epoch <= pred_err_epoch_max:
        print("pred_err_loop:%d, err_num:%d, err_map:%s, current_lr：%s" % (pred_err_epoch, len(pred_err_dataset), pred_err_map, str(current_lr)))
        pred_err_epoch += 1
        pred_err_map = [0, 0, 0, 0, 0, 0, 0]
        train_loss = 0
        correct = 0
        total = 0
        time_start = time.time()
        batch_idx = 0
        del pred_err_dataset_temp
        pred_err_dataset_temp = pred_err_dataset
        pred_err_dataset = RawDataSet()
        pred_err_loader = torch.utils.data.DataLoader(pred_err_dataset_temp, batch_size=opt.bs, shuffle=True)
        pred_err_prefetcher = Prefetcher(pred_err_loader)
        inputs, targets = pred_err_prefetcher.next()
        while inputs is not None:
            optimizer.zero_grad()
            outputs, targets = put_through_net(inputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            utils.clip_gradient(optimizer, 2*current_lr)
            optimizer.step()

            train_loss += float(loss.data)
            _, predicted = torch.max(outputs.data, 1)
            if target_type == 'ls':
                ground_value = targets.data
            elif target_type == 'fa':
                _, ground_value = torch.max(targets.data, 1)

            for i in range(len(predicted)):
                if predicted[i] != ground_value[i]:
                    pred_err_dataset.add(inputs[i], targets[i])
                    pred_err_map[ground_value[i].item()] += 1

            total += targets.size(0)
            correct += predicted.eq(ground_value.data).cpu().sum()
            cur_train_acc = float(correct) / float(total) * 100.

            time_end = time.time()
            duration = time_end - time_start
            utils.progress_bar(batch_idx, len(pred_err_loader), 'Time: %.2fs | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                               (duration, train_loss / (batch_idx + 1), cur_train_acc, correct, total))

            # 删除无用的变量，释放显存
            del loss
            del inputs
            del outputs
            del predicted

            inputs, targets = pred_err_prefetcher.next()
            batch_idx += 1
    del pred_err_dataset
    del pred_err_dataset_temp


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


def write_history(train_or_test, epoch, acc, loss, predictions):
    '''
    将数据写入history.txt文件保存
    :param train_or_test: 训练过程还是测试过程（'Train' or 'Test'）
    :param epoch: 迭代次数
    :param acc: 准确率
    :param loss: 损失
    :param predictions: 预测情况
    :return: 无
    '''
    with open(os.path.join(net_to_save_path, history_file_name), "a+", encoding="utf-8") as history_file:
        msg = train_or_test + " %d %.3f %.3f " % (epoch, acc, loss)
        if predictions:
            msg += str(predictions)
        msg += "\n"
        history_file.write(msg)
        history_file.flush()


def save_over_flag():
    '''
    创建一个空文件表示训练完成
    :return: 无
    '''
    file_path = os.path.join(net_to_save_path, model_over_flag_name)
    with open(file_path, "w+", encoding="utf-8") as file:
        file.write(train_acc_map.__str__())
        file.write("\n")
        file.write(test_acc_map.__str__())
        file.write("\n")
        file.flush()


if __name__ == "__main__":
    if not os.path.isdir(net_to_save_dir):
        os.mkdir(net_to_save_dir)
    if not os.path.isdir(os.path.join(net_to_save_dir, str(opt.save_number))):
        os.mkdir(os.path.join(net_to_save_dir, str(opt.save_number)))
    if not os.path.isdir(net_to_save_path):
        os.mkdir(net_to_save_path)
    train_prefetcher = None
    test_prefetcher = None
    if not over_flag:
        for epoch in range(start_epoch, opt.epoch, 1):
            print('\n------------Epoch: %d-------------' % epoch)
            if use_cuda:
                train_prefetcher = Prefetcher(train_loader)
                test_prefetcher = Prefetcher(test_loader)
                train(epoch)
                test(epoch)
            else:
                old_train(epoch)
                old_test(epoch)
            temp_internal -= 1
            if temp_internal <= 0:
                temp_internal = TEMP_EPOCH
                print("Saving Temp Model...")
                state = {'net': net.state_dict() if use_cuda else net,
                         'best_test_acc': test_acc_map['best_acc'],
                         'best_test_acc_epoch': test_acc_map['best_acc_epoch'],
                         'best_train_acc': train_acc_map['best_acc'],
                         'best_train_acc_epoch': train_acc_map['best_acc_epoch'],
                         'cur_epoch': epoch,
                         }
                torch.save(state, os.path.join(net_to_save_path, saved_temp_model_name))
        print(train_acc_map)
        print(test_acc_map)
        save_over_flag()
    print("Trained Over")
