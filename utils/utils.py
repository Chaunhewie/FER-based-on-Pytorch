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
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def draw_pic(arr):
    img = Image.fromarray(arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(img), cmap="gray")
    plt.show()