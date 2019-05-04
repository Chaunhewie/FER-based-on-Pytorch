# coding=utf-8
import os
import math
import torch
import torch.nn as nn
'''
假设输入为 W*W 大小的图片
核 Kernal = F*F
步长 Stride = S
填充 padding = P
输出图片的大小为 N*N  N = (W-F+2P)/S + 1
                    W = (N-1)*S-2P+F
感受野
RF = 1 #待计算的feature map上的感受野大小
for layer in （top layer To down layer）:
    RF = ((RF - 1)* stride) + fsize  # 此处padding区域不考虑，其他和上述公式相同，只是RF刚开始变成1
                    
论文：120*120 crop->96*96 as inputs

'''

class ACCNN(nn.Module):
    '''ACCNN 自己创建的神经网络，用于识别面部表情

        n_class 表示输出的分类数
        pre_trained 表示是否加载训练好的网络
        root_pre_path 表示执行目录相对于项目目录的路径（用于debug）
        dataset 表示预加载使用的模型参数训练自哪个数据集
        fold 表示存储的文件序号
        virtualize 表示是否进行可视化
        using_fl 表示是否为根据face landmarks进行识别
    '''
    def __init__(self, n_classes=7, pre_trained=False, root_pre_path='', dataset='FER2013', fold=5, virtualize=False,
                 using_fl=False):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACCNN, self).__init__()
        self.input_size = 96
        # x size [BATCHSIZE, 1, 120, 120]
        # 测试论文
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),   # 64, 92, 92
        #     nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),   # 64, 88, 88
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),      # 64, 44, 44
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 64, 40, 40
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 64, 36, 36
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=1),      # 64, 18, 18
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(64 * 18 * 18, 64),
        #     nn.Dropout(p=0.6),
        #     nn.Linear(64, n_classes),
        #     nn.Dropout(p=0.6),
        #     nn.Softmax(1),
        # )
        # 1 # 40
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, dilation=2),  # 32, 88, 88
        #     nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, dilation=2),  # 32, 80, 80
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),  # 32, 76, 76
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),  # 32, 38, 38
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),  # 32, 34, 34
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),  # 32, 30, 30
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),  # 32, 15, 15
        # )
        # self.classifier = nn.Sequential(
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(32 * 15 * 15, 128),
        #     nn.Dropout(p=0.6),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(128, n_classes),
        #     nn.Dropout(p=0.6),
        #     nn.Softmax(1),
        # )
        # 2 # 55
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, dilation=2),  # 64, 88, 88
#             nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),  # 64, 84, 84
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),  # 64, 40, 40
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),  # 64, 20, 20
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),  # 64, 16, 16
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),  # 64, 12, 12
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),  # 64, 6, 6
#         )
#         self.classifier = nn.Sequential(
#             # nn.ReLU(inplace=True),
#             nn.Linear(64 * 6 * 6, 256),
#             nn.Dropout(p=0.6),
#             # nn.ReLU(inplace=True),
#             nn.Linear(256, n_classes),
#             nn.Dropout(p=0.6),
#             nn.Softmax(1),
#         )
        # 3 # 99
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, dilation=2),  # 64, 88, 88
#             nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),  # 64, 42, 42
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),  # 64, 38, 38
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),  # 64, 19, 19
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),  # 64, 19, 19
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),  # 64, 19, 19
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),  # 64, 9, 9
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64, 9, 9
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64, 9, 9
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),  # 64, 4, 4
#         )
#         self.classifier = nn.Sequential(
#             # nn.ReLU(inplace=True),
#             nn.Linear(64 * 4 * 4, 512),
#             nn.Dropout(p=0.6),
#             # nn.ReLU(inplace=True),
#             nn.Linear(512, n_classes),
#             nn.Dropout(p=0.6),
#             nn.Softmax(1),
#         )
        # 4 # 72
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, dilation=2),  # 64, 88, 88
            nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64, 86, 86
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64, 84, 84
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64, 82, 82
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 64, 81, 81
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),  # 64, 40, 40
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),  # 64, 20, 20
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64, 10, 10
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64, 8, 8
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64, 6, 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 64, 5, 5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.ReLU(inplace=True),
            nn.Linear(64 * 5 * 5, 512),
            nn.Dropout(p=0.6),
            nn.Linear(512, n_classes),
            nn.Softmax(1),
        )
        self.dataset = dataset
        self.fold = fold
        self.virtualize = virtualize
        self.using_fl = using_fl
        self.features_out = []
        self.best_acc = 0.
        self.best_acc_epoch = -1
        if dataset == "CK+" or dataset == "CK+48":
            self.output_map = {0:'生气', 1:'蔑视', 2:'恶心', 3:'害怕', 4:'开心', 5:'悲伤', 6:'惊讶'}
        elif dataset == 'FER2013':
            self.output_map = {0:'生气', 1:'恶心', 2:'害怕', 3:'开心', 4:'悲伤', 5:'惊讶', 6:'中性'}
        elif dataset == 'JAFFE':
            self.output_map = {0:'中性', 1:'开心', 2:'悲伤', 3:'惊讶', 4:'生气', 5:'恶心', 6:'害怕'}
        else:
            assert 'dataset error: should be in ["JAFFE", "CK+48", "CK+", "FER2013"]'
            self.output_map = {}
        if pre_trained:
            save_model_dir_name = 'Saved_Models'
            if self.using_fl:
                saved_model_name = "Best_model_fl.t7"
            else:
                saved_model_name = 'Best_model.t7'
            net_saved_path = os.path.join(root_pre_path, save_model_dir_name, str(fold), dataset + '_ACCNN_' + str(fold))
            assert os.path.isdir(net_saved_path), 'Error: no checkpoint directory found!'
            parameters_file_path = os.path.join(net_saved_path, saved_model_name)
            print("Loading parameters from ", parameters_file_path)
            checkpoint = torch.load(parameters_file_path)
            self.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['best_test_acc']
            self.best_acc_epoch = checkpoint['best_test_acc_epoch']
            print("Loading parameters over!")
            print("Parameters are trained from %s(epoch %d) with test_acc: %3.f"
                  % (dataset, self.best_acc_epoch, self.best_acc))
        else:
            print('Initializing ACCNN weights...')
            self._initialize_weights()
        print('Init ACCNN model over!')

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

    def clean_features_out(self):
        self.features_out = []

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
    net = ACCNN(n_classes=n_classes, root_pre_path='..', pre_trained=False)
    print(net)
    print("num_of_parameters_of_net: ", num_of_parameters_of_net(net))
