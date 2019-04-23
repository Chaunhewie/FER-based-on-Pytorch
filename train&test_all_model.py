# coding=utf-8
import os
enabled_nets = ["ACNN", "ACCNN", "AlexNet", "VGG11", "VGG13", "VGG16", "VGG19"]
enabled_datasets = ["JAFFE", "CK+", "FER2013"]

if __name__ == "__main__":
    for dataset in enabled_datasets:
        for net in enabled_nets:
            command = "python train_test.py --dataset %s --model %s" % (dataset, net)
            print(command)
            os.system(command)

# python train_test.py --dataset JAFFE --model AlexNet