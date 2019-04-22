# coding=utf-8
import os
import face_recognition
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

img_save_dir="Saved_Virtualizations"




if __name__ == "__main__":
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
