# coding=utf-8
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, n_classes=1000):
        super(AlexNet, self).__init__()
        self.input_size = 223
        # N = (W-F+2P)/S + 1   W = (N-1)*S-2P+F
        # INPUT 223, 223
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 55, 55    W = (N-1)*S-2P+F = 26*2-0+3 = 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27, 27    W = (N-1)*S-2P+F = 26*1-4+5 = 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 27, 27    W = (N-1)*S-2P+F = 12*2-0+3 = 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13, 13    W = (N-1)*S-2P+F = 12*1-2+3 = 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  #
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        # print("step 0:", x.size())
        x = self.features(x)
        # print("step 1:", x.size())
        self.features_out = x.clone()
        x = x.view(x.size(0), 256 * 6 * 6)
        # print("step 2:", x.size())
        x = self.classifier(x)
        # print("step 3:", x.size())
        return x

if __name__ == "__main__":
    n_classes = 7
    net = AlexNet(n_classes=n_classes)
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
