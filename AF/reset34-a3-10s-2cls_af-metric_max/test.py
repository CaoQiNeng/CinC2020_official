import numpy as np

a = np.array([[0,2,0],[2,0,0],[2,0,0]])

b = np.argmax(a,axis=0)
for i, index in enumerate(b):
    a[i][index] = 1

print(a)