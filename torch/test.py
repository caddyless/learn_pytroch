import numpy as np
import math
import matplotlib.pylab as plt
import random


def get_img():
    im = plt.imread('beauty.jpg')
    shape = im.shape
    minmal = min(shape[:1])
    im = im[:minmal, :minmal]
    return im, minmal


def create(x):
    if len(x) != 3:
        print('参数错误!')
        return
    str_x = 0
    for i in range(3):
        str_x = str_x * 1000 + x[i]
    return str_x


def parse(str_x):
    x = np.zeros(3)
    for i in range(3):
        x[2 - i] = int(str_x % 1000)
        str_x = int(str_x / 1000)
    return x


def img_show(im):
    plt.imshow(im)
    plt.show()


def img_change(im, edge,seed=[]):
    if len(seed)==0:
        seed = np.arange(edge)
        random.shuffle(seed)
    iden = np.zeros((edge, edge))
    for i in range(edge):
        iden[i][seed[i]] = 1
    new_im = np.zeros(im.shape[:2])
    new_img=np.zeros(im.shape)
    for i in range(edge):
        for j in range(edge):
            new_im[i][j] = create(im[i][j])
    new_im = iden.dot(new_im)
    for i in range(edge):
        for j in range(edge):
            for k in range(3):
                new_img[i][j][k] = parse(new_im[i][j])[k]
    return new_img,seed


if __name__ == '__main__':
    im, minmal = get_img()
    new,seed = img_change(im, minmal)
    newer=img_change(new,minmal,seed)
    img_show(newer)
