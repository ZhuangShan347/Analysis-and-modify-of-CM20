import numpy as np
from PRF import prf
import os
import pickle
import matplotlib.pyplot as plt

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
    if (r < 30) or (r == 30):
        x = np.array([2, 1])
    elif (r>30) and (r<55):
        x = np.array([2, 2])
    elif (r>54) and (r<80):
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


Aly0 = []
Aly1 = []
Aly2 = []
Aly3 = []
for n in range(1000):
  for i in range(n):
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
  for j in fenxi.keys():
      if list(j) == [3, 6]:
          Aly0.append(fenxi[j]) 
      elif list(j) == [6, 6]:
          Aly1.append(fenxi[j])
      elif list(j) == [2, 6]:
          Aly2.append(fenxi[j])
      elif list(j) == [5, 6]:
          Aly3.append(fenxi[j])
      #print(Aly1)
   
  #Aly.append(fenxi)
    

fig = plt.figure()
an = plt.subplot(111) 

plt.xlabel('Number of iterations')
plt.ylabel('Error')
an.plot(Aly0, label='(3, 6)')
an.plot(Aly1, label='(6, 6)')
an.plot(Aly2, label='(2, 6)')
an.plot(Aly3, label='(5, 6)')
plt.legend()
plt.grid(True)
#plt.plot(k0)
plt.show()
