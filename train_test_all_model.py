# coding=utf-8
import os
enabled_nets = ["ACNN", "ACCNN", "AlexNet", "VGG11", "VGG19", "ResNet18", "ResNet50", "ResNet152"]
enabled_datasets = ["JAFFE", "CK+", "FER2013"]
train_batch_size = {"ACNN": 32, "ACCNN": 32, "AlexNet": 32, "VGG11": 32, "VGG19": 4, "ResNet18": 32, "ResNet50": 32, "ResNet152": 12}

if __name__ == "__main__":
    for dataset in enabled_datasets:
        for net in enabled_nets:
            bs = train_batch_size[net]
            command = "python train_test.py --dataset %s --model %s --bs %s" % (dataset, net, bs)
            print(command)
            os.system(command)

# python train_test.py --dataset JAFFE --model AlexNet
