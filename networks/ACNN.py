# coding=utf-8
import math
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
    def __init__(self, n_classes, virtualize=False):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACNN, self).__init__()
        self.input_size = 48
        # x size [BATCHSIZE, 1, 48, 48]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),   # 6, 44, 44
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 6, 22, 22
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5),  # 16, 18, 18
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 16, 9, 9
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),  # 64, 9, 9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # 64, 4, 4
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, padding=1),  # 10, 4, 4
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),      # 10, 2, 2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(10 * 2 * 2, 16),
            nn.Dropout(),
            nn.Linear(16, n_classes),
            nn.Softmax(1),
        )
        self.virtualize = virtualize
        self.features_out = []
        print('Initializing ACNN weights...')
        self._initialize_weights()

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        if self.virtualize:
            self.features_out.append(x.clone())
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

    def _initialize_weights(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.Conv2d):
                n = layer[1].kernel_size[0] * layer[1].kernel_size[1] * layer[1].out_channels
                layer[1].weight.data.normal_(0, math.sqrt(2. / n))
                if layer[1].bias is not None:
                    layer[1].bias.data.zero_()
            elif isinstance(layer[1], nn.Linear):
                layer[1].weight.data.normal_(0, 0.01)
                layer[1].bias.data.zero_()


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.utils import num_of_parameters_of_net
    n_classes = 7
    net = ACNN(n_classes=n_classes)
    print(net)
    print("num_of_parameters_of_net: ", num_of_parameters_of_net(net))
