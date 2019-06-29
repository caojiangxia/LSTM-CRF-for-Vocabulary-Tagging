import numpy as np

A=np.array([1,2,3])
B=np.array([4,5,3])
print(np.in1d(A,B).sum())

B=np.array([[4,5,3],[3,4,5]])
print(len(B.reshape(-1)))