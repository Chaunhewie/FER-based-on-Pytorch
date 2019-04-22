# coding=utf-8
import os
import torch
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
    def __init__(self, n_classes, pre_trained=False, root_pre_path='', data_set='FER2013', fold=2):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(ACCNN, self).__init__()
        self.input_size = 223
        # x size [BATCHSIZE, 1, 48, 48]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),   # 32, 110, 110
            nn.ReLU(inplace=True),  # 使用nn.ReLU(inplace = True) 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中，节省不少显存
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
            nn.Softmax(1),
        )
        self.features_out = []
        self.best_acc = 0.
        self.best_acc_epoch = -1
        if data_set == "CK+" or data_set == "CK+48":
            self.output_map = {0:'生气', 1:'蔑视', 2:'恶心', 3:'害怕', 4:'开心', 5:'悲伤', 6:'惊讶'}
        elif data_set == 'FER2013':
            self.output_map = {0:'生气', 1:'恶心', 2:'害怕', 3:'开心', 4:'悲伤', 5:'惊讶', 6:'中性'}
        elif data_set == 'JAFFE':
            self.output_map = {0:'中性', 1:'开心', 2:'悲伤', 3:'惊讶', 4:'生气', 5:'恶心', 6:'害怕'}
        else:
            assert 'dataset error: should be in ["JAFFE", "CK+48", "CK+", "FER2013"]'
            self.output_map = {}
        if pre_trained:
            save_model_dir_name = 'Saved_Models'
            saved_model_name = 'Best_model.t7'
            net_to_save_path = os.path.join(root_pre_path, save_model_dir_name, data_set+"_ACCNN_"+str(fold))
            print("Loading parameters from ", net_to_save_path)
            assert os.path.isdir(net_to_save_path), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join(net_to_save_path, saved_model_name))
            self.load_state_dict(checkpoint['net'])
            print("Loading parameters over!")
            self.best_acc = checkpoint['best_test_acc']
            self.best_acc_epoch = checkpoint['best_test_acc_epoch']
        print('Init ACCNN model over!')

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
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


if __name__ == "__main__":
    n_classes = 7
    net = ACCNN(n_classes=n_classes, root_pre_path='..', pre_trained=True)
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