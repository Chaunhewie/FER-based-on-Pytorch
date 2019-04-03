# coding=utf-8
import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import argparse

from networks.ACNN import ACNN
from networks.AlexNet import AlexNet
from dal.JAFFE_DataSet import JAFFE
import transforms.transforms
import utils.utils as utils

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

parser = argparse.ArgumentParser(description='PyTorch CNN Training With JAFFE')
# parser.add_argument('--model', type=str, default='ACNN', help='CNN architecture')
parser.add_argument('--model', type=str, default='AlexNet', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='JAFFE', help='dataset')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=1, type=int, help='training epoch num')
parser.add_argument('--lrd_se', default=500, type=int, help='learning rate decay start epoch')  # 表示默认从epoch为500开始进行lr的递减
parser.add_argument('--lrd_s', default=20, type=int, help='learning rate decay step')  # 表示默认每经过2次epoch进行一次递减
parser.add_argument('--lrd_r', default=0.9, type=float, help='learning rate decay rate')  # 表示每次的lr的递减率，默认每递减一次乘一次0.9
opt = parser.parse_args()

n_classes = 7
if opt.model == "ACNN":
    net = ACNN(n_classes=n_classes).to(DEVICE)
elif opt.model == "AlexNet":
    net = AlexNet(n_classes=n_classes).to(DEVICE)

input_img_size = net.input_size
transform_train = transforms.Compose([
    transforms.Resize(input_img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_img_size = int(input_img_size * 1.1)  # 测试时，图片resize大小
transform_test = transforms.Compose([
    transforms.Resize(test_img_size),
    transforms.TenCrop(input_img_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# criterion = nn.MSELoss()
# target_type = 'fa'
criterion = nn.CrossEntropyLoss()
target_type = 'ls'
# 随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

train_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
test_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

print("------------Preparing Data...----------------")
train_data = JAFFE(is_train=True, transform=transform_train, target_type=target_type)
test_data = JAFFE(is_train=False, transform=transform_test, target_type=target_type)
print("------------Data Already be Prepared---------")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.bs, shuffle=False)


# Training
def train(epoch):
    print('\n------------Epoch: %d-------------' % epoch)
    # 根据训练的epoch次数来降低learning rate
    if epoch > opt.lrd_se > 0:
        frac = (epoch - opt.lrd_se) // opt.lrd_s
        decay_factor = opt.lrd_r ** frac
        current_lr = opt.lr * decay_factor
        # current_lr = opt.lr * 降低率 ^ ((epoch - 开始decay的epoch) // 每次decay的epoch num)
        utils.set_lr(optimizer, current_lr)  # set the learning rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    cur_train_acc = 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE, torch.long)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        # print("outputs:", outputs)
        # print("targets:", targets)
        loss = criterion(outputs, targets)
        loss.backward()
        # utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.data
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
        cur_train_acc = (100. * correct / total).item()

        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                           (train_loss / (batch_idx + 1), cur_train_acc, correct, total))

    Train_acc = cur_train_acc
    if train_acc_map['best_acc'] < Train_acc:
        train_acc_map['best_acc'] = Train_acc
        train_acc_map['best_acc_epoch'] = epoch


# Testing
def test(epoch):
    global Test_acc
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    cur_test_acc = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)

            if use_cuda:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
            if target_type == 'ls':
                ground_value = targets.data
            elif target_type == 'fa':
                _, ground_value = torch.max(targets.data, 1)

            for i in range(len(predicted)):
                if predicted[i] == ground_value[i]:
                    test_acc_map[predicted[i].item()] += 1

            total += targets.size(0)
            correct += predicted.eq(ground_value.data).cpu().sum()
            cur_test_acc = (100. * correct / total).item()

            utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (PrivateTest_loss / (batch_idx + 1), cur_test_acc, correct, total))

    Test_acc = cur_test_acc
    if test_acc_map['best_acc'] < Test_acc:
        test_acc_map['best_acc'] = Test_acc
        test_acc_map['best_acc_epoch'] = epoch


if __name__ == "__main__":
    for epoch in range(opt.epoch):
        train(epoch)
        # for name,parameters in net.named_parameters():
        #     print(name,':',parameters.size())
        #     print(parameters)
        #     break
        for parameters in net.parameters():
            print(parameters.size())
            print(parameters[0][0][0])
            break
        test(epoch)
    print(train_acc_map)
    print(test_acc_map)
