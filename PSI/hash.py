import numpy as np
from PRF import prf
import os
import pickle
import matplotlib.pyplot as plt
import datetime

def H1(A, x, q):
    c = np.dot(A, x) % q
    return c 

def generate_or_load_matrix(filename, shape):
    matrix = np.random.randint(0, q, size=shape)
     
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

Times1 = []

Times2 = []

n = 200

psi = []

prg = []

for i in range(n):

    starttime =datetime.datetime.now()
    q = i + 1 
    n1 = i
    #w = 20

    # k = np.array([1, 0, 1, 1, 0])
    # x = np.array([2, 3, 1, 4])

    k = np.random.randint(0, 2, size=i+1)

    #print(k)

    x = np.random.randint(0, q, size=i)

    # n1 = len(x)
    # print(len(x))
    m = n1 + 1

    A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
    A1 = generate_or_load_matrix(A1_filename, (m, n1))

    c = H1(A1, x, q)

    # Generate or load A2 matrix
    A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
    A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

    v = prf(A2, c, k, q)

    D = [[1] * len(c) for _ in range(len(c))]

    #print(len(v))   

    for j in range(len(c)):

        D[j][v[j]]=0 
    
    endtime = datetime.datetime.now()
     
    tt0=endtime-starttime #jishi    

    starttime =datetime.datetime.now()

    for l in range(len(c)):

        psi.append(hash(tuple(D[l])))
    #print(psi)
    endtime = datetime.datetime.now()
     
    tt1=endtime-starttime #jishi

    Times1.append((tt0+tt1).microseconds/10000)

    ###########################################################################
    ###########################################################################

    starttime =datetime.datetime.now()

    for l in range(len(c)):

        A3 = np.random.randint(0, 2, size=(len(c),len(c)))

        e = np.random.randint(0, 2, size=len(c))

        psi_=  np.dot(A3, np.array(D[l])) + e 

        prg.append(psi_)
    #print(psi)
    endtime = datetime.datetime.now()
     
    tt2=endtime-starttime #jishi

    Times2.append((tt0+tt2).microseconds/10000)
    

fig = plt.figure()
an = plt.subplot(111) 

plt.xlabel('n')
plt.ylabel('CPU Time')
an.plot(Times1, label='CM20')
an.plot(Times2, label='Ours')

plt.legend()

#plt.plot(k0)
plt.show()




# for i in range(10):
#     x = xcel()
#     n1 = len(x)

#     # Generate or load A1 matrix
#     A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
#     A1 = generate_or_load_matrix(A1_filename, (m, n1))

#     c = H1(A1, x, q)

#     # Generate or load A2 matrix
#     A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
#     A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

#     v = prf(A2, c, k, q)

#     Va.append(v)

#     #V_dict[np.array(v)] = x

# fenxi = calculate_proportions(Va)

# for item, proportion in fenxi.items():
#     print(f"元素 {item} 的比例为: {proportion:.2f}%")

# #print(Va,fenxi)
    
# XX = [np.array([2, 1]), np.array([2, 2]), np.array([2, 3]), np.array([2, 4])]

# for x in XX:
#    A1_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A1_matrix.pkl'
#    A1 = generate_or_load_matrix(A1_filename, (m, n1))

#    c = H1(A1, x, q)

#    # Generate or load A2 matrix
#    A2_filename = 'E:/单壮/学期成果1/第四学期/PSI分析/PSI/A2_matrix.pkl'
#    A2 = generate_or_load_matrix(A2_filename, (len(c) + 1, len(c)))

#    v = prf(A2, c, k, q)

#    print(x,v)