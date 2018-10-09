from entry import get_parameter
import numpy as np
import torch
import torchvision as tv
from get_image import download_img

eigenvalue = np.array([1027.9272, 483.76813, 342.99887,
                       61.083946, 30.049044, 7.57872])
eigenvector = np.array([[0.05512872, 0.7531318, 0.2113893, -0.51538473, -0.3265917, -0.11306769],
                        [0.22430435, 0.25530243, -0.21368574, -
                            0.19913605, 0.87150097, -0.19919206],
                        [0.26042345, -0.08233467, -0.29312116, -
                            0.3865124, -0.01324988, 0.83061016],
                        [0.5385896, 0.21776046, -0.6222087,
                            0.36599708, -0.31742418, -0.20160851],
                        [-0.25589696, -0.38590908, -0.4868617, -
                            0.5938197, -0.1696001, -0.40886503],
                        [-0.72337127, 0.40557212, -0.44671574, 0.2423961, 0.06423409, 0.22317792]])
# energy=[]
# for i in range(len(eigenvalue)):
#     energy.append(eigenvalue[:i+1].sum()/eigenvalue.sum())
# Ud_pie=eigenvector[:3].T
# M=Ud_pie.dot(Ud_pie.T)
# print(eigenvector-M)
# print(energy)
# print(eigenvector.mean(0))
sorted_indices = np.argsort(eigenvalue)
print(eigenvalue)
print(eigenvalue[-1:-1])
