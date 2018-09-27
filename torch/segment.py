import numpy as np
import matplotlib.pylab as plt
import random
import os
import turicreate as tc
import scipy.misc as mc

rootdir='/home/qualcomm/101_ObjectCategories'
dstdir='/home/qualcomm/seg_images'

def get_img():
    folder_list=os.listdir(rootdir)
    for folder in folder_list:
        sour_path=os.path.join(rootdir,folder)
        dst_path=os.path.join(dst_path,folder)
        image_list=os.listdir(sour_path)
        for img in image_list:
            im = plt.imread(os.path.join(sour_path,im))
            shape = im.shape
            minmal = min(shape[:2])
            im = im[:minmal, :minmal]
            new_im,seed=img_change(im,minmal)
            assert new_im.shape[2]==3
            if not os.path.isdir(dst_path):
                os.makedirs(dst_path)
            mc.imsave(os.path.join(dst_path, img))
    return


def create(x):
    if len(x) != 3:
        print('参数错误!')
        return
    str_x = 0
    for i in range(3):
        str_x = str_x * 256 + x[i]
    return str_x


def parse(str_x):
    x = np.zeros(3)
    for i in range(3):
        x[2 - i] = int(str_x % 256)
        str_x = int(str_x / 256)
    return x


def img_show(img):
    seat=121
    for im in img:
        plt.figure('compare')
        plt.subplot(seat)
        plt.imshow(im)
        seat=seat+1
    plt.show()


def generate_seed(edge):
    seed = np.arange(edge)
    random.shuffle(seed)
    return seed


def img_change(im, edge, seed=[]):
    if len(seed) == 0:
        seed=generate_seed(edge)
    iden = np.zeros((edge, edge))
    for i in range(edge):
        iden[i][seed[i]] = 1
    new_im = np.zeros(im.shape[:2])
    new_img = np.zeros(im.shape)
    for i in range(edge):
        for j in range(edge):
            new_im[i][j] = create(im[i][j])
    new_im = iden.dot(new_im)
    for i in range(edge):
        for j in range(edge):
            for k in range(3):
                new_img[i][j][k] = parse(new_im[i][j])[k]
    img=new_img.astype(np.uint8)
    return img, seed


if __name__ == '__main__':
    get_img()