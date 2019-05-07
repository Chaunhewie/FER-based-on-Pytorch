# coding=utf-8
import os
import face_recognition
from PIL import Image
import numpy as np
from matplotlib.lines import Line2D

img_save_dir="Saved_Virtualizations"


def crop_face_area_and_get_landmarks(img, image_size=400):
    """
    人脸定位并进行人脸的面部剪裁
    :param img: 原图
    :param image_size: 剪裁后的图片大小设置
    :return: 剪裁后的图片，面部区域在原图中的位置，面部标记在剪裁后的图中的位置
    """
    # 图片转化为灰度图
    img = img.convert("L")
    # 获取图片的人脸定位
    face_locations = face_recognition.face_locations(np.array(img))  # fer2013存在无法识别的图片
    if len(face_locations) <= 0:
        return img, None, None
    face_box = face_locations[0]
    top, right, bottom, left = face_box
    # 脸部关键点标记获取
    face_landmarks = face_recognition.face_landmarks(np.array(img))
    if len(face_landmarks) <= 0:
        return img, None, None
    face_landmarks = face_landmarks[0]
    # 扩充脸部区域
    for name, plot_list in face_landmarks.items():
        for plot in plot_list:
            if plot[0] < left:
                left = plot[0] if plot[0] > 0 else 0
            if plot[0] > right:
                right = plot[0] if plot[0] > 0 else 0
            if plot[1] < top:
                top = plot[1] if plot[1] > 0 else 0
            if plot[1] > bottom:
                bottom = plot[1] if plot[1] > 0 else 0
    top_exp = int((bottom-top)*0.1)
    bottom_exp = int((bottom-top)*0.05)
    left_exp, right_exp = int((right-left)*0.1), int((right-left)*0.1)
    # 脸部剪裁
    crop_loc = (left-left_exp, top-top_exp, right+right_exp, bottom+bottom_exp)
    img = img.crop(crop_loc).resize((image_size, image_size))  # , Image.ANTIALIAS
    # 更新面部标记点的位置
    face_landmarks = face_recognition.face_landmarks(np.array(img))
    if len(face_landmarks) <= 0:
        return img, None, None
    face_landmarks = face_landmarks[0]
    return img, face_box, face_landmarks


def get_img_with_landmarks(img, landmarks, round_to_keep=15):
    """
    根据face landmarks来剪裁img，得到标记点附近的img图像
    :param img: 原img
    :param landmarks: 标记点
    :param inplace: 是否直接操作原img
    :param round_to_keep: 保留标记点周围的 round_to_keep 个像素点
    :return: 剪裁后的img
    """
    point_map = {}
    for name, points in landmarks.items():
        for point in points:
            for x_bias in range(-round_to_keep, round_to_keep+1, 1):
                for y_bias in range(-round_to_keep, round_to_keep+1, 1):
                    x, y = point[0]+x_bias, point[1]+y_bias
                    if x not in point_map:
                        point_map[x] = {}
                    point_map[x][y] = True
    img_arr = np.array(img)
    row_num, col_num = img_arr.shape[0], img_arr.shape[1]
    for y in range(row_num):
        for x in range(col_num):
            if x in point_map and y in point_map[x]:
                continue
            else:
                img_arr[y][x] = 0.
    return Image.fromarray(img_arr)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_dir_pre_path = "E:\毕设\代码\my_scripts\data\CK+"
    classes = os.listdir(img_dir_pre_path)
    for c in classes:
        img_file_names = os.listdir(os.path.join(img_dir_pre_path, c))
        for img_file_name in img_file_names:
            img = Image.open(os.path.join(img_dir_pre_path, c, img_file_name))
            break
        img = np.array(img)
        top, right, bottom, left = face_recognition.face_locations(img)[0]
        face_landmarks = face_recognition.face_landmarks(img)[0]
        for name, plot_list in face_landmarks.items():
            for plot in plot_list:
                if plot[0] < left:
                    left = plot[0]
                if plot[0] > right:
                    right = plot[0]
                if plot[1] < top:
                    top = plot[1]
                if plot[1] > bottom:
                    bottom = plot[1]
        region = img.crop()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.array(img), cmap="gray")

        for name, plot_list in face_landmarks.items():
            for plot in plot_list:
                ax.plot(plot[0], plot[1], 'bo')
        lines = []
        lines.append([(top, top), (left, right)])
        lines.append([(top, bottom), (left, left)])
        lines.append([(top, bottom), (right, right)])
        lines.append([(bottom, bottom), (left, right)])
        for line_ys, line_xs in lines:
            #         print(line_xs,line_ys)
            ax.add_line(Line2D(line_xs, line_ys, linewidth=2, color='blue'))
        plt.savefig(os.path.join(img_save_dir, "CK+_" + c))
    plt.close('all')
