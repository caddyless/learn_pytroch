import numpy as np
import ITQ

Y = np.arange(24)
Y = Y.reshape(4, 6)
A = Y.dot(Y.T)
eivalues, eivector = np.linalg.eig(A)

sorted_indices = np.argsort(-eivalues)
eivalues = np.array([eivalues[item] for item in sorted_indices])
eivector = np.array([eivector[:, item] for item in sorted_indices])
index = 0
for i in range(len(eivalues)):
    ratio = eivalues[:(i+1)].sum()/eivalues.sum()
    if ratio > 0.95:
        index = i+1
        break
Ud_pie = eivector[:index]
print(Ud_pie)
test_eigen = eivector[0].reshape((4, 1))
print(test_eigen)
print(index)
print(eivector.shape)
print(test_eigen.shape)
print(A.dot(test_eigen))
print(eivalues[0]*test_eigen)
print(A)
print(test_eigen.dot(test_eigen.T)*eivalues[0])
