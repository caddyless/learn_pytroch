import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_img():
    I=cv2.imread('beauty.jpg')
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    I=np.array(I)
    return I

def show_imgs(img_list):
    plt.figure('compare')
    num=0
    for img in img_list:
        plt.subplot(231)
        plt.imshow(img, cmap='gray')
        plt.show()

def svd(array):
    U,sigma,Vt=np.linalg.svd(array,full_matrices=False)
    print(sigma)


if __name__ == '__main__':
    img=get_img()
    svd(img)


