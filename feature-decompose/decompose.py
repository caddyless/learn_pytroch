import torch
import numpy as np
import torchvision as tv
from entry import get_parameter

par = get_parameter()


def get_M(samples):
    YYT = samples.dot(samples.T)
    eigenvalue, eigenvector = np.linalg.eig(YYT)
    print('特征值为:' + str(eigenvalue))
    print('特征向量为:' + str(eigenvector))
    return


def feature_decompose(samples, w, bias):
    sy1 = np.array(samples)
    sy1 = sy1.reshape(par['c2'], -1)
    get_M(sy1)
    return


if __name__ == '__main__':
    feature_decompose()
