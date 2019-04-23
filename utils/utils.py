# coding=utf-8
'''
Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Total_Bar_Length = 30


def progress_bar(cur, tot, msg):
    s = "\r["
    prog = int(float(Total_Bar_Length) * (cur + 1) / tot)
    rest = Total_Bar_Length - prog - 1
    s = s + "=" * prog + ">" + "." * rest + "]"
    s += " | " + msg
    if cur < tot - 1:
        print(s, end="")
    else:
        print(s)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def draw_img(img, save_path="", plt_show=True, log_enabled=True):
    """
    绘制图像
    :param img: img
    :return: None
    """
    fig = plt.figure(figsize=(20, 20))  # figsize: width, height in inches
    ax = fig.add_subplot(111)
    ax.imshow(np.array(img), cmap="gray")
    if len(save_path) > 0:
        plt.savefig(save_path)
        if log_enabled:
            print("saved fig to %s" % save_path)
    if plt_show:
        plt.show()
    else:
        plt.close('all')


def num_of_parameters_of_net(net):
    num_of_parameters = 0
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        # print(parameters)
        num = 1
        for i in parameters.size():
            num *= i
        print(num)
        num_of_parameters += num
    return num_of_parameters
