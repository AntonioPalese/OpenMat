import numpy as np
N = 4
a = np.zeros(shape=(N, N)) + 2.0;
b = np.zeros(shape=(N, N)) + 1.0;

print(a@b)