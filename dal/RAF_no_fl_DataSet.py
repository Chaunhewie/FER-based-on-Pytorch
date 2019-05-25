# coding=utf-8
import os
from PIL import Image
import numpy as np
import torch.utils.data as data


class RAF_no_fl(data.Dataset):
    """`RAF_no_fl Dataset.
    Args:
        is_train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
                                        E.g, ``transforms.RandomCrop``
        target_type (str, optional): Using for target type: "fa" for "float array", "ls" for "long single".
                                    E.g, ``MSELoss will use fa``; ``CrossEntropyLoss will use ls``
        img_dir_pre_path (str, optional): The relative path of the data dictionary and main file.

        Before using the dataset, 'data/RAF_no_fl' dictionary must be build(running RAF_DataSet with 'self.save_img_no_fl=True')
        The dataset contains aligned images of RAF which can't get face landmarks with face_recognition package.
    """

    def __init__(self, is_train=True, transform=None, target_type="ls", img_dir_pre_path="data/RAF_no_fl",
                 RAF_img_dir_pre_path="data/RAF"):
        if target_type == "fa":
            self.classes_map = {0: np.array([1., 0., 0., 0., 0., 0., 0.], dtype=float),
                                1: np.array([0., 1., 0., 0., 0., 0., 0.], dtype=float),
                                2: np.array([0., 0., 1., 0., 0., 0., 0.], dtype=float),
                                3: np.array([0., 0., 0., 1., 0., 0., 0.], dtype=float),
                                4: np.array([0., 0., 0., 0., 1., 0., 0.], dtype=float),
                                5: np.array([0., 0., 0., 0., 0., 1., 0.], dtype=float),
                                6: np.array([0., 0., 0., 0., 0., 0., 1.], dtype=float)}
        elif target_type == "ls":
            self.classes_map = {0: 0,
                                1: 1,
                                2: 2,
                                3: 3,
                                4: 4,
                                5: 5,
                                6: 6}
        else:
            assert ("target_type should be 'fa' or 'ls', but input is %s" % (target_type))
        self.img_aligned_data_dir_path = os.path.join(img_dir_pre_path)
        self.cla_data_file_path = os.path.join(RAF_img_dir_pre_path, 'EmoLabel', 'list_patition_label.txt')
        self.transform = transform
        self.is_train = is_train  # train set or test set
        self.name = 'RAF_no_fl'

        self.cla_map = {}
        self.cla_num_list = [0, 0, 0, 0, 0, 0, 0]
        self.get_cla_map()

        print("正在处理图片数据...")
        self.train_data = []
        self.train_data_num = 0
        self.train_classes = []
        self.test_data = []
        self.test_data_num = 0
        self.test_classes = []
        img_file_names = os.listdir(self.img_aligned_data_dir_path)
        for img_file_name in img_file_names:
            file_type, file_index = img_file_name.strip().split('.')[0].split("_")
            if file_type == 'train':
                self.train_data_num += 1
                if is_train:
                    img = Image.open(os.path.join(self.img_aligned_data_dir_path, img_file_name)).convert("L")
                    self.train_data.append(img)
                    self.train_classes.append(self.classes_map[self.cla_map[file_type + "_" + file_index]])
            elif file_type == 'test':
                self.test_data_num += 1
                if not is_train:
                    img = Image.open(os.path.join(self.img_aligned_data_dir_path, img_file_name)).convert("L")
                    self.test_data.append(img)
                    self.test_classes.append(self.classes_map[self.cla_map[file_type + "_" + file_index]])
        print("处理图片数据完成！")
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

    def set_transform(self, transform):
        self.transform = transform

    def get_cla_map(self):
        '''
        读取label分类信息
        :return: 无
        '''
        with open(self.cla_data_file_path) as file:
            lines = file.readlines()
        # 将分类信息处理成map
        for line in lines:
            file_name, classification = line.strip().split(" ")
            file_name = file_name.split(".")[0]
            #     print(file_name, classification)
            self.cla_map[file_name] = int(classification) - 1
            self.cla_num_list[int(classification) - 1] += 1
        print("分类信息读取完毕！")
        print("数据集已分类的图片个数：", len(self.cla_map))
        print("已分类各类图片个数：", self.cla_num_list)


if __name__ == "__main__":
    f1 = RAF_no_fl(is_train=True, img_dir_pre_path="../data/RAF_no_fl", RAF_img_dir_pre_path="../data/RAF")
    f2 = RAF_no_fl(is_train=False, img_dir_pre_path="../data/RAF_no_fl", RAF_img_dir_pre_path="../data/RAF")
    print(f1.__len__(), f2.__len__())

    from utils.utils import draw_img

    draw_img(f1.__getitem__(0)[0])
    draw_img(f2.__getitem__(0)[0])
