# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import random
import os
import scipy.misc as mc
import math

rootdir = '/home/qualcomm/101_ObjectCategories'
dstdir = '/home/qualcomm/seg_images'


def get_img(fraction=100):
    ser = np.arange(fraction)
    random.shuffle(ser)
    my_dstdir = dstdir + '/fraction+' + str(fraction)
    folder_list = os.listdir(rootdir)
    for folder in folder_list:
        sour_path = os.path.join(rootdir, folder)
        dst_path = os.path.join(my_dstdir, folder)
        image_list = os.listdir(sour_path)
        for img in image_list:
            im = plt.imread(os.path.join(sour_path, img))
            shape = im.shape
            if len(shape) != 3:
                print(folder + ' ' + img + 'abandoned')
                continue
            print(folder + ' ' + img)
            minmal = min(shape[:2])
            im = im[:minmal, :minmal]
            seed = generate_seed(shape, ser, fraction)
            new_im = img_change(im, shape, seed)
            assert new_im.shape[2] == 3
            if not os.path.isdir(dst_path):
                os.makedirs(dst_path)
            mc.imsave(os.path.join(dst_path, img))
    return


# 将三维矩阵压缩为二维矩阵
def create(x):
    if len(x) != 3:
        print('参数错误!')
        return
    str_x = 0
    for i in range(3):
        str_x = str_x * 256 + x[i]
    return str_x


# 将压缩的二维矩阵还原为三维矩阵
def parse(str_x):
    x = np.zeros(3)
    for i in range(3):
        x[2 - i] = int(str_x % 256)
        str_x = int(str_x / 256)
    return x


def img_show(img):
    seat = 121
    for im in img:
        plt.figure('compare')
        plt.subplot(seat)
        plt.imshow(im)
        seat = seat + 1
    plt.show()


# 根据给出的份数fractions给出行变换矩阵的1值坐标
def generate_seed(shape, ser, fractions=100):
    assert len(shape) == 3
    f = fractions
    granularity = math.floor(shape[0] / f)
    g = granularity
    m = f - shape[0] % f
    n = shape[0] % f
    assert m * g + n * (g + 1) == shape[0]
    seed = np.zeros((shape[0]))
    index = 0
    for item in ser:
        if item < m:
            k = g
            start = g * item
        elif item >= m and item < f:
            k = g + 1
            start = g * m + (g + 1) * (item - m)
        else:
            print('item越界')
            assert 0
        if start > shape[0]:
            print('start越界')
            assert 0
        for i in range(k):
            seed[index] = start + i
            index += 1
            if index >= shape[0]:
                break
    return seed


def img_change(im, shape, seed):
    assert len(shape) == 3
    if len(seed) == 0:
        seed = generate_seed(shape)
    iden = np.zeros((shape[0], shape[0]))
    for i in range(shape[0]):
        iden[i][seed[i]] = 1
    new_im = np.zeros(im.shape[:2])
    new_img = np.zeros(im.shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_im[i][j] = create(im[i][j])
    new_im = iden.dot(new_im)
    assert shape[2] == 3
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(3):
                new_img[i][j][k] = parse(new_im[i][j])[k]
    img = new_img.astype(np.uint8)
    return img


if __name__ == '__main__':
    get_img()
