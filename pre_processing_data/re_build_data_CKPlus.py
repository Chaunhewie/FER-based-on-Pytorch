# coding=utf-8
'''
本文件用于将原始的CK+图片数据集分成八大类：
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
'''

import os

original_data_dir_path = "../data/CK+/cohn-kanade-images"
rebuild_dir_path = "../data/CK+/my_data"



def rebuild_data():
    dirs = os.listdir(original_data_dir_path)
    dirs2 = os.listdir(os.path.join(original_data_dir_path, dirs[0]))
    print(dirs2)



if __name__ == "__main__":
    rebuild_data()