# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F


class ACNN(nn.Module):
    def __init__(self, n_classes):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACNN, self).__init__()

        # kernel
        # 1 input image channel, 10 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 10, 5)
        # 10 input image channel, 10 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(10, 10, 5)
        # 10 input image channel, 10 output channels, 3x3 square convolution
        self.conv3 = nn.Conv2d(10, 10, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, n_classes)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # x size [BATCHSIZE, 1, 48, 48]
        # print("0:", x.size())
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2))  # x size [BATCHSIZE, 10, 22, 22]
        # print("1:", x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # x size [BATCHSIZE, 10, 9, 9]
        # print("2:", x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # x size [BATCHSIZE, 10, 3, 3]
        # print("3:", x.size())
        # 这里做的就是压扁的操作 就是把后面的[BATCHSIZE, 10, 3, 3]压扁，变为 [4, 90]
        x = x.view(-1, self.num_flat_features(x))
        # print("3:", x.size())
        # 输入为 144
        x = F.relu(self.fc1(x))  # x size [BATCHSIZE, 64]
        # print("11:", x.size())
        x = F.relu(self.fc2(x))  # x size [BATCHSIZE, 16]
        # print("12:", x.size())
        x = self.fc3(x)  # x size [BATCHSIZE, n_classes]
        # print("13:", x.size())
        x = self.softmax(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    input_img_size = 48
    n_classes = 7
    net = ACNN(n_classes=n_classes)
    print(net)

    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        print(parameters)
