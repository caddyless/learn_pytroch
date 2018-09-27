import numpy as np
from numpy import *
import matplotlib.pyplot as plt


mat=np.ones((3,3))
U,sigma,Vt=np.linalg.svd(mat)
print(U)
print(sigma)
print(Vt)