# coding=utf-8
import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from networks.ACNN import ACNN
from dal.JAFFE_DataSet import JAFFE
import transforms.transforms
import utils.utils as utils

BATCH_SIZE=128
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
learning_rate = 0.01

input_img_size = 48
n_classes = 7
net = ACNN(n_classes=n_classes).to(DEVICE)

train_img_size = 64
transform_train = transforms.Compose([
    transforms.Resize(input_img_size),
#     transforms.TenCrop(input_img_size),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_img_size = 64
transform_test = transforms.Compose([
    transforms.Resize(test_img_size),
    transforms.TenCrop(input_img_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

train_data = JAFFE(True, transform_train)
test_data = JAFFE(False, transform_test)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
# 随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
test_acc_map = {'best_acc': 0, 'best_acc_epoch': -1, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

start_epoch = 0
total_epoch = 1000


# Training
def train(epoch):
    print('\n------------Epoch: %d-------------' % epoch)
    print('learning_rate: %s' % str(learning_rate))
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    cur_train_acc = 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE, torch.float)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        #         print(outputs)
        #         print(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        #         utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)  # torch.max() 加上dim参数后，返回值为 max_value, max_value_index
        _, ground_value = torch.max(targets.data, 1)

        for i in range(len(predicted)):
            if predicted[i] == ground_value[i]:
                train_acc_map[predicted[i].item()] += 1

        total += targets.size(0)
        correct += predicted.eq(ground_value.data).cpu().sum()
        cur_train_acc = (100. * correct / total).item()

        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), cur_train_acc, correct, total))

    Train_acc = cur_train_acc
    if train_acc_map['best_acc'] < Train_acc:
        train_acc_map['best_acc'] = Train_acc
        train_acc_map['best_acc_epoch'] = epoch


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
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE, torch.float)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
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
    for epoch in range(start_epoch, total_epoch):
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
