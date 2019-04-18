# coding=utf-8
import os
import torch.nn as nn
'''
假设输入为 W*W 大小的图片
核 Kernal = F*F
步长 Stride = S
填充 padding = P
输出图片的大小为 N*N  N = (W-F+2P)/S + 1
                    W = (N-1)*S-2P+F
'''

class ACCNN(nn.Module):
    def __init__(self, n_classes):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACCNN, self).__init__()
        self.input_size = 223
        # x size [BATCHSIZE, 1, 48, 48]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),   # 32, 110, 110
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),   # 32, 53, 53
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),   # 32, 24, 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 32, 12, 12
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),  # 16, 10, 10
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),  # 16, 10, 10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),      # 16, 5, 5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(16 * 5 * 5, 20),
            nn.Dropout(p=0.5),
            nn.Linear(20, n_classes),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        self.features_out = x.clone()
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size())
        x = self.classifier(x)
        # print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    n_classes = 7
    net = ACCNN(n_classes=n_classes)
    print(net)

    num_of_parameters = 0
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        # print(parameters)
        num = 1
        for i in parameters.size():
            num *= i
        print(num)
        num_of_parameters += num
    print(num_of_parameters)