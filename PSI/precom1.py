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

def xcel():
    r = np.random.randint(0,100)
    if (r < 32) or (r == 32):
        x = np.array([2, 1])
    elif (r>32) and (r<60):
        x = np.array([2, 2])
    elif (r>59) and (r<82):
        x = np.array([2, 3])
    else:
        x = np.array([2, 4])
    return x


def calculate_proportions(lst):
    counts = {}
    proportions = {}
    total = len(lst)

    for item in lst:
        if tuple(item) in counts:
            counts[tuple(item)] += 1
        else:
            counts[tuple(item)] = 1

    for item, count in counts.items():
        proportion = (count / total) * 100
        proportions[item] = proportion

    return proportions




q = 7 

k = np.array([1, 0])
m = 2

Va = []
V_dict = {}


x = np.array([2, 3])







for i in range(10):
    x = xcel()
    n1 = len(x)

    # Generate or load A1 matrix
    A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
    A1 = generate_or_load_matrix(A1_filename, (m, n1))

    c = H1(A1, x, q)

    # Generate or load A2 matrix
    A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
    A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

    v = prf(A2, c, k, q)

    Va.append(v)

    #V_dict[np.array(v)] = x

fenxi = calculate_proportions(Va)

for item, proportion in fenxi.items():
    print(f"元素 {item} 的比例为: {proportion:.2f}%")

#print(Va,fenxi)
    
XX = [np.array([2, 1]), np.array([2, 2]), np.array([2, 3]), np.array([2, 4])]

for x in XX:
   A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
   A1 = generate_or_load_matrix(A1_filename, (m, n1))

   c = H1(A1, x, q)

   # Generate or load A2 matrix
   A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
   A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

   v = prf(A2, c, k, q)

   print(x,v)