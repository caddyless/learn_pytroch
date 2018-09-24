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
    print(x)
    return x


def img_show(im):
    plt.imshow(im)
    plt.show()


def img_change(im, edge):
    ser = np.arange(edge)
    random.shuffle(ser)
    iden = np.zeros((edge, edge))
    for i in range(edge):
        iden[i][ser[i]] = 1
    new_im = np.zeros(im.shape[:1])
    for i in range(edge):
        for j in range(edge):
            new_im[i][j] = create(im[i][j])
    new_im = iden * new_im
    for i in range(edge):
        for j in range(edge):
            for k in range(3):
                im[i][j][k] = parse(new_im[i][j])[k]
    return im


if __name__ == '__main__':
    im, minmal = get_img()

    im = img_change(im, minmal)
    img_show(im)
