# coding=utf-8
import torch.nn as nn

'''
假设输入为 W*W 大小的图片
核 Kernal = F*F
步长 Stride = S
填充 padding = P
输出图片的大小为 N*N  N = (W-F+2P)/S + 1
                    W = (N-1)*S-2P+F
'''

class ACNN(nn.Module):
    def __init__(self, n_classes):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACNN, self).__init__()
        self.input_size = 48
        # x size [BATCHSIZE, 1, 48, 48]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),   # 10, 44, 44
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),      # 10, 22, 22
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5),  # 10, 18, 18
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 10, 9, 9
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),  # 10, 7, 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 10, 3, 3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(10 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(16, n_classes),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
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
    net = ACNN(n_classes=n_classes)
    print(net)

    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        print(parameters)
