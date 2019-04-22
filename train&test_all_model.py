# coding=utf-8
import os
enabled_nets = ["ACNN", "ACCNN", "AlexNet"]
enabled_datasets = ["JAFFE", "CK+48", "CK+", "FER2013"]

if __name__ == "__main__":
    for dataset in enabled_datasets:
        for net in enabled_nets:
            command = "python train_test.py --dataset %s --model %s" % (dataset, net)
            print(command)
            os.system(command)

# python train_test.py --dataset JAFFE --model AlexNet