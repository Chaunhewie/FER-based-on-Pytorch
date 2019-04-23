# coding=utf-8
import os
enabled_nets = ["ACCNN", "AlexNet", "VGG11", "VGG19", "ResNet18", "ResNet50", "ResNet152", "ACNN"]
enabled_datasets = ["CK+", "FER2013", "JAFFE"]
train_batch_size = {"ACNN": 32, "ACCNN": 32, "AlexNet": 32, "VGG11": 12, "VGG19": 4, "ResNet18": 32, "ResNet50": 32, "ResNet152": 12}


if __name__ == "__main__":
    begin_index = 2
    for dataset in enabled_datasets:
        for net in enabled_nets:
            if begin_index > 0:
                begin_index -= 1
                continue
            bs = train_batch_size[net]
            command = "python train_test.py --dataset %s --model %s --bs %s" % (dataset, net, bs)
            print(command)
            os.system(command)

# python train_test.py --dataset JAFFE --model AlexNet
