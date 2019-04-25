# coding=utf-8
import os
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
import sys
sys.path.append('..')
from utils.face_recognition import crop_face_area_and_get_landmarks, get_img_with_landmarks

All_People_Indexes = ['S119', 'S130', 'S127', 'S073', 'S067', 'S107', 'S092', 'S160', 'S134', 'S106', 'S101', 'S155',
                      'S109', 'S053', 'S116', 'S139', 'S064', 'S117', 'S505', 'S099', 'S122', 'S082', 'S079', 'S121',
                      'S066', 'S115', 'S102', 'S050', 'S131', 'S071', 'S093', 'S077', 'S062', 'S055', 'S011', 'S034',
                      'S506', 'S014', 'S074', 'S087', 'S059', 'S010', 'S029', 'S091', 'S058', 'S113', 'S085', 'S136',
                      'S097', 'S068', 'S061', 'S114', 'S057', 'S045', 'S149', 'S156', 'S054', 'S051', 'S108', 'S110',
                      'S105', 'S135', 'S999', 'S151', 'S504', 'S026', 'S083', 'S502', 'S503', 'S028', 'S100', 'S005',
                      'S158', 'S078', 'S060', 'S080', 'S133', 'S112', 'S501', 'S065', 'S075', 'S069', 'S096', 'S124',
                      'S147', 'S137', 'S063', 'S132', 'S128', 'S022', 'S037', 'S086', 'S138', 'S046', 'S032', 'S120',
                      'S125', 'S094', 'S042', 'S072', 'S070', 'S052', 'S089', 'S076', 'S088', 'S148', 'S090', 'S056',
                      'S118', 'S129', 'S103', 'S095', 'S084', 'S154', 'S157', 'S104', 'S111', 'S035', 'S044', 'S098',
                      'S126', 'S895', 'S081']
# random.shuffle(All_People_Indexes)
# print(All_People_Indexes)


class CKPlus(data.Dataset):
    """`CK+ Dataset & CK+48 Dataset, CK+48 is aborted after adding face location detect and crop operation.
    Args:
        is_train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
                                        E.g, ``transforms.RandomCrop``
        target_type(str, optional): Using for target type: "fa" for "float array", "ls" for "long single"
                                    E.g, ``MSELoss will use fa``; ``CrossEntropyLoss will use ls``
        k_folder (int, optional): Using for split the dataset as train set and test set,
                                  and len(test set):len(train set) = 1:(10-k_folder)
        img_dir_pre_path (str, optional): The relative path of the data dictionary and main file

        there are 981(anger:135 contempt:54 disgust:177 fear:75 happy:207 sadness:84 surprise:249) images in data with 123 people
        we choose images of 111 people, whose name is in self.train_people_names, for training
        we choose images of 12 person, whose name is in self.test_people_names, for testing
    """

    def __init__(self, is_train=True, transform=None, target_type="fa", k_folder=1, img_dir_pre_path="data/CK+", using_fl=False):
        if target_type == "fa":
            self.classes_map = {'anger': np.array([1., 0., 0., 0., 0., 0., 0.], dtype=float),
                                'contempt': np.array([0., 1., 0., 0., 0., 0., 0.], dtype=float),
                                'disgust': np.array([0., 0., 1., 0., 0., 0., 0.], dtype=float),
                                'fear': np.array([0., 0., 0., 1., 0., 0., 0.], dtype=float),
                                'happy': np.array([0., 0., 0., 0., 1., 0., 0.], dtype=float),
                                'sadness': np.array([0., 0., 0., 0., 0., 1., 0.], dtype=float),
                                'surprise': np.array([0., 0., 0., 0., 0., 0., 1.], dtype=float)}
        elif target_type == "ls":
            self.classes_map = {'anger': 0,
                                'contempt': 1,
                                'disgust': 2,
                                'fear': 3,
                                'happy': 4,
                                'sadness': 5,
                                'surprise': 6}
        else:
            assert("target_type should be 'fa' or 'ls', but input is %s" % (target_type))
        self.img_dir_pre_path = img_dir_pre_path
        self.transform = transform
        self.is_train = is_train  # train set or test set
        self.using_fl = using_fl

        split_index = int(len(All_People_Indexes)*k_folder/10)
        if split_index < 1:
            split_index = 1
        self.train_people_indexes = All_People_Indexes[:-split_index]
        self.test_people_indexes = All_People_Indexes[-split_index:]
        # print(self.train_people_indexes, self.test_people_indexes)

        self.train_data = []
        self.train_data_num = 0
        self.train_classes = []
        self.test_data = []
        self.test_data_num = 0
        self.test_classes = []
        classes = os.listdir(self.img_dir_pre_path)
        for c in classes:
            img_file_names = os.listdir(os.path.join(self.img_dir_pre_path, c))
            for img_file_name in img_file_names:
                if img_file_name[:4] in self.train_people_indexes:
                    self.train_data_num += 1
                    if is_train:
                        img = Image.open(os.path.join(self.img_dir_pre_path, c, img_file_name))
                        img, face_box, face_landmarks = crop_face_area_and_get_landmarks(img)
                        if face_box is None or face_landmarks is None:
                            self.train_data_num -= 1
                            continue
                        if using_fl:
                            landmarks_img = get_img_with_landmarks(img, face_landmarks)
                            self.train_data.append(landmarks_img)
                        else:
                            self.train_data.append(img)
                        self.train_classes.append(self.classes_map[c])
                elif img_file_name[:4] in self.test_people_indexes:
                    self.test_data_num += 1
                    if not is_train:
                        img = Image.open(os.path.join(self.img_dir_pre_path, c, img_file_name))
                        img, face_box, face_landmarks = crop_face_area_and_get_landmarks(img)
                        if face_box is None or face_landmarks is None:
                            self.train_data_num -= 1
                            continue
                        if using_fl:
                            landmarks_img = get_img_with_landmarks(img, face_landmarks)
                            self.test_data.append(landmarks_img)
                        else:
                            self.test_data.append(img)
                        self.test_classes.append(self.classes_map[c])
                else:
                    print("img:(%s,%s) is not belong to both of train or test set!" % (c, img_file_name))
        print("train_num: ", self.train_data_num, " test_num:", self.test_data_num)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index >= self.__len__():
            return None, None, None, None

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


if __name__ == "__main__":
    c1 = CKPlus(is_train=True, img_dir_pre_path="../data/CK+")
    c2 = CKPlus(is_train=False, img_dir_pre_path="../data/CK+")
    print(c1.__len__(), c2.__len__())

    from utils.utils import draw_img

    draw_img(c1.__getitem__(0)[0])
    draw_img(c2.__getitem__(0)[0])
