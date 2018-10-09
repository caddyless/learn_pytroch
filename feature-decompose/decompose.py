import torch
import numpy as np
import torchvision as tv
from entry import get_parameter

par = get_parameter()


def get_M(samples):
    YYT = samples.dot(samples.T)
    eigenvalue, eigenvector = np.linalg.eig(YYT)
    sorted_indices = np.argsort(eigenvalue)
    print('特征值为:' + str(eigenvalue))
    print('特征向量为:' + str(eigenvector))
    energy = []
    d_pie = 0
    for i in range(len(eigenvalue)):
        energy.append(eigenvalue[:i+1].sum() / eigenvalue.sum())
        if energy[i] > 0.94:
            d_pie = i+1
            break

    print('能量为:'+str(energy))
    Ud_pie = eigenvector[:, sorted_indices[:-d_pie - 1:-1]]
    print('前'+str(d_pie)+'个特征向量为:' + str(Ud_pie))
    M = Ud_pie.dot(Ud_pie.T)
    print('矩阵M为:'+str(M))
    return Ud_pie, d_pie


def feature_decompose(samples, w, bias,channel):
    print('权重shape='+str(w.shape))
    samples = np.array(samples)
    print('样本形状为：'+str(samples.shape))
    samples = np.transpose(samples, (3, 0, 1, 2))
    samples = samples.reshape((channel, -1), order='F')
    samples = samples-samples.mean(0)
    Ud_pie, d_pie = get_M(samples)
    print('矩阵为:'+str(Ud_pie))
    return Ud_pie


if __name__ == '__main__':
    feature_decompose()
