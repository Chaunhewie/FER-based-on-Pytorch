# coding=utf-8
import os
import random
from PIL import Image
import numpy as np
import torch.utils.data as data

All_People_Names = ['KM', 'KL', 'KR', 'YM', 'UY', 'NA', 'NM', 'MK', 'KA', 'TM']
# random.shuffle(All_People_Names)
# print(All_People_Names)


class JAFFE(data.Dataset):
    """`JAFFE Dataset.
    Args:
        is_train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
                                        E.g, ``transforms.RandomCrop``
        target_type (str, optional): Using for target type: "fa" for "float array", "ls" for "long single"
                                    E.g, ``MSELoss will use fa``; ``CrossEntropyLoss will use ls``
        k_folder (int, optional): Using for split the dataset as train set and test set,
                                  and len(test set):len(train set) = 1:(10-k_folder)
        img_dir_pre_path (str, optional): The relative path of the data dictionary and main file

        there are 213(NEU:30 HAP:31 SAD:31 SUR:30 ANG:30 DIS:29 FEA:32) images in data with ten people
        we choose images of 9 people, whose name is in self.train_people_names, for training
        we choose images of 1 person, whose name is in self.test_people_names, for testing
    """

    def __init__(self, is_train=True, transform=None, target_type="fa", k_folder=1, img_dir_pre_path="data/jaffe"):
        if target_type == "fa":
            self.classes_map = {'NE': np.array([1., 0., 0., 0., 0., 0., 0.], dtype=float),
                                'HA': np.array([0., 1., 0., 0., 0., 0., 0.], dtype=float),
                                'SA': np.array([0., 0., 1., 0., 0., 0., 0.], dtype=float),
                                'SU': np.array([0., 0., 0., 1., 0., 0., 0.], dtype=float),
                                'AN': np.array([0., 0., 0., 0., 1., 0., 0.], dtype=float),
                                'DI': np.array([0., 0., 0., 0., 0., 1., 0.], dtype=float),
                                'FE': np.array([0., 0., 0., 0., 0., 0., 1.], dtype=float)}
        elif target_type == "ls":
            self.classes_map = {'NE': 0,
                                'HA': 1,
                                'SA': 2,
                                'SU': 3,
                                'AN': 4,
                                'DI': 5,
                                'FE': 6}
        else:
            assert("target_type should be 'fa' or 'ls', but input is %s" % (target_type))
        self.img_dir_pre_path = img_dir_pre_path
        self.transform = transform
        self.is_train = is_train  # train set or test set

        split_index = int(len(All_People_Names)*k_folder/10)
        if split_index < 1:
            split_index = 1
        self.train_people_names = All_People_Names[:-split_index]
        self.test_people_names = All_People_Names[-split_index:]
        # print(self.train_people_names, self.test_people_names)

        self.train_data = []
        self.train_data_num = 0
        self.train_classes = []
        self.test_data = []
        self.test_data_num = 0
        self.test_classes = []
        for person_name in self.train_people_names:
            img_file_names = os.listdir(os.path.join(self.img_dir_pre_path, person_name))
            self.train_data_num += len(img_file_names)
            if is_train:
                for img_file_name in img_file_names:
                    img = Image.open(os.path.join(self.img_dir_pre_path, person_name, img_file_name))
                    self.train_data.append(img)
                    self.train_classes.append(self.classes_map[img_file_name[3:5]])
        for person_name in self.test_people_names:
            img_file_names = os.listdir(os.path.join(self.img_dir_pre_path, person_name))
            self.test_data_num += len(img_file_names)
            if not is_train:
                for img_file_name in img_file_names:
                    img = Image.open(os.path.join(self.img_dir_pre_path, person_name, img_file_name))
                    self.test_data.append(img)
                    self.test_classes.append(self.classes_map[img_file_name[3:5]])
        print("train_num: ", self.train_data_num, " test_num:", self.test_data_num)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index >= self.__len__():
            return None, None
        if self.is_train:
            img, cla = self.train_data[index], self.train_classes[index]
        else:
            img, cla = self.test_data[index], self.test_classes[index]

        img = img.convert("L")
        # 由于存在 random_crop 等的随机处理，应该是读取的时候进行，这样每个epoch都能够获取不同的random处理
        if self.transform is not None:
            img = self.transform(img)

        return img, cla

    def __len__(self):
        """
        Returns:
            int: data num.
        """
        if self.is_train:
            return self.train_data_num
        else:
            return self.test_data_num


if __name__ == "__main__":
    j1 = JAFFE(is_train=True, img_dir_pre_path="../data/jaffe")
    j2 = JAFFE(is_train=False, img_dir_pre_path="../data/jaffe")
    print(j1.__len__(), j2.__len__())

    from utils.utils import draw_pic
    draw_pic(j1.__getitem__(0)[0])
    draw_pic(j2.__getitem__(0)[0])
