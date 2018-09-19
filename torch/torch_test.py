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
    series=231
    for img in img_list:
        plt.subplot(series)
        plt.imshow(img, cmap='gray')
        series=series+1
        num=num+1
    plt.show()


if __name__ == '__main__':
    img=get_img()
    U,sigma,Vt=np.linalg.svd(img,full_matrices=False)
    img_list=[]
    img_list.append(img)
    sig=np.zeros(sigma.shape)
    for i in range(1,101,20):
        for j in range(i):
            sig[j]=sigma[j]
        new_img=np.dot(U*sig,Vt)
        img_list.append(new_img)
    show_imgs(img_list)



