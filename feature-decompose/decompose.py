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
    energy = []
    for i in range(len(eigenvalue)):
        energy.append(eigenvalue[:i + 1].sum() / eigenvalue.sum())
    Ud_pie = eigenvector[:3].T
    M = Ud_pie.dot(Ud_pie.T)
    return M


def feature_decompose(samples, w, bias,channel):
    samples=samples.reshape(channel,-1)
    samples=samples-samples.mean(0)
    M=get_M(samples)
    return M


if __name__ == '__main__':
    feature_decompose()
