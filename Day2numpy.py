import numpy as np

A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(A)
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
print(B)
C = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
x = np.transpose(B)
print(x)
y = np.linalg.inv(B)
print(y)
D = B + C 
print(D)
M = np.dot(B,C)
print(M)