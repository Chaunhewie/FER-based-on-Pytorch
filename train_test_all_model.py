# coding=utf-8
import os
enabled_nets = ["ACCNN", "AlexNet", "VGG11", "VGG19", "ResNet18", "ResNet50", "ResNet152", "ACNN"]
enabled_datasets = ["FER2013", "CK+", "JAFFE"]
train_epoch = {"ACNN": 500, "ACCNN": 500, "AlexNet": 300, "VGG11": 200, "VGG19": 200, "ResNet18": 200, "ResNet50": 200,
               "ResNet152": 200}

# 使用自己的电脑
train_batch_size = {"ACNN": 128, "ACCNN": 128, "AlexNet": 128, "VGG11": 12, "VGG19": 4, "ResNet18": 32, "ResNet50": 24,
                    "ResNet152": 12}
# 阿里云服务器
# train_batch_size = {"ACNN": 128, "ACCNN": 128, "AlexNet": 128, "VGG11": 64, "VGG19": 32, "ResNet18": 64, "ResNet50": 48,
#                     "ResNet152": 32}

if __name__ == "__main__":
    # begin_index = 0
    for dataset in enabled_datasets:
        for net in enabled_nets:
            # if begin_index > 0:
            #     begin_index -= 1
            #     continue
            bs = train_batch_size[net]
            epoch = train_epoch[net]
            lrd_se = int(epoch*0.8)
            lrd_s = int((epoch-lrd_se)/10)
            command = "python train_test.py --dataset %s --model %s --bs %s --epoch %d --lrd_se %d --lrd_s %d" \
                      % (dataset, net, bs, epoch, lrd_se, lrd_s)
            print(command)
            os.system(command)

# python train_test.py --dataset JAFFE --model AlexNet
