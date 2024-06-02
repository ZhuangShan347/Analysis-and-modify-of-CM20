import numpy as np
from PRF import prf
import os
import pickle

def H1(A, x, q):
    c = np.dot(A, x) % q
    return c 

def generate_or_load_matrix(filename, shape):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            matrix = pickle.load(f)
    else:
        matrix = np.random.randint(0, q, size=shape)
        with open(filename, 'wb') as f:
            pickle.dump(matrix, f)
    return matrix

q = 7 
x = np.array([2, 7])
k = np.array([1, 0])
m = 2
n1 = len(x)

# Generate or load A1 matrix
A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
A1 = generate_or_load_matrix(A1_filename, (m, n1))

c = H1(A1, x, q)

# Generate or load A2 matrix
A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

v = prf(A2, c, k, q)

print(x, v)