import numpy as np
from PRF import prf
import os
import pickle
import matplotlib.pyplot as plt
import datetime

def H1(A, x, q):
    c = np.dot(A, x) % q
    return c 

def generate_or_load_matrix(shape,q):
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


def dec_CM12(n):

    CM12 = []

    for i in range(n):

        q = 21 
        n1 = q-1
        #w = 20

        # k = np.array([1, 0, 1, 1, 0])
        # x = np.array([2, 3, 1, 4])

        k = np.random.randint(0, 2, size=q)

        #print(k)

        x = np.random.randint(0, q, size=n1)

        # n1 = len(x)
        # print(len(x))
        m = n1 + 1

        A1 = generate_or_load_matrix((m, n1),q)

        c = H1(A1, x, q)

        # Generate or load A2 matrix

        A2 = generate_or_load_matrix((len(c) + 1, len(c)),q)

        v = prf(A2, c, k, q)

        D = [[1] * len(c) for _ in range(len(c))]

        #print(len(v))   

        for j in range(len(c)):

            D[j][v[j]]=0 
        

        psi = []
        for l in range(len(c)):

            psi.append(hash(tuple(D[l])))

            #print(psi)

        CM12.append(psi)

        #print(CM12)

    a = np.random.randint(0, n)

    scelet_a = CM12[a]

    #print('CM12',CM12[a])

    starttime =datetime.datetime.now()

    for i in  CM12:

        if i ==  scelet_a:

            break

    endtime = datetime.datetime.now()
        
    tt1=endtime-starttime #jishi

    return tt1


def dec_Ours(n):

    Ours1 = []

    Ours2 = []

    for i in range(n):

        q = 21 
        n1 = q-1
        #w = 20

        # k = np.array([1, 0, 1, 1, 0])
        # x = np.array([2, 3, 1, 4])

        k = np.random.randint(0, 2, size=q)

        #print(k)

        x = np.random.randint(0, q, size=n1)

        # n1 = len(x)
        # print(len(x))
        m = n1 + 1

        A1 = generate_or_load_matrix((m, n1),q)

        c = H1(A1, x, q)

        # Generate or load A2 matrix

        A2 = generate_or_load_matrix((len(c) + 1, len(c)),q)

        v = prf(A2, c, k, q)

        D = [[1] * len(c) for _ in range(len(c))]

        #print(len(v))   

        for j in range(len(c)):

            D[j][v[j]]=0 
        #print(D)

        prg1 = []

        A3 = np.random.randint(0, 2, size=(len(c),len(c)))

        for l in range(len(c)):
                        

            e = np.random.randint(0, 2, size=len(c))

            #print('e',e)

            psi1=  np.dot(A3, np.array(D[l])) + e 

            prg1.append(list(psi1))

            #print(prg)

            
        Ours1.append(prg1)

        prg2 = []


        for l in range(len(c)):

            #A3 = np.random.randint(0, 2, size=(len(c),len(c)))

            e = np.random.randint(0, 2, size=len(c))

            psi2=  np.dot(A3, np.array(D[l])) + e 

            prg2.append(list(psi2))

            #print(prg)

        Ours2.append(prg2)
        #print(psi)
          

    a = np.random.randint(0, n)

    scelet_a = Ours1[a]

    #print('scelet_a',scelet_a)

    #print('CM12',Ours1)

    starttime =datetime.datetime.now()


    for i in  Ours2:
        
        error = []
   
        for sublist1, sublist2 in zip(i, scelet_a):
            temp = [x - y for x, y in zip(sublist1, sublist2)]
            error.append(temp)
        #print(result)
        #result = [1 if x in [-1, 1, 0] else 0 for x in result]
        #print('result',error)  

        for j in error:

            if all(((_ == 0) or (_ == 1) or (_ == -1)) for _ in j):

                #print('i',i)

                break

    endtime = datetime.datetime.now()
    
    tt2=endtime-starttime #jishi

    return tt2



Times1 = []

Times2 = []

n = 200
for i in range(1,n):
    tt1 = dec_CM12(i)
    tt2 = dec_Ours(i)

    Times1.append(tt1.microseconds)

    Times2.append(tt2.microseconds)







   
    # ###########################################################################
    # ###########################################################################

    # starttime =datetime.datetime.now()

    # for l in range(len(c)):

    #     A3 = np.random.randint(0, 2, size=(len(c),len(c)))

    #     e = np.random.randint(0, 2, size=len(c))

    #     psi_=  np.dot(A3, np.array(v)) + e 

    #     prg.append(psi_)
    # #print(psi)
    # endtime = datetime.datetime.now()
     
    # tt2=endtime-starttime #jishi

    # Times2.append((tt0+tt2).microseconds/10000)










# for i in range(n):

#     starttime =datetime.datetime.now()
#     q = i + 1 
#     n1 = i
#     #w = 20

#     # k = np.array([1, 0, 1, 1, 0])
#     # x = np.array([2, 3, 1, 4])

#     k = np.random.randint(0, 2, size=i+1)

#     #print(k)

#     x = np.random.randint(0, q, size=i)

#     # n1 = len(x)
#     # print(len(x))
#     m = n1 + 1

#     A1 = generate_or_load_matrix((m, n1))

#     c = H1(A1, x, q)

#     # Generate or load A2 matrix

#     A2 = generate_or_load_matrix((len(c) + 1, len(c)))

#     v = prf(A2, c, k, q)

#     D = [[1] * len(c) for _ in range(len(c))]

#     #print(len(v))   

#     for j in range(len(c)):

#         D[j][v[j]]=0 
    
#     endtime = datetime.datetime.now()
     
#     tt0=endtime-starttime #jishi    

#     starttime =datetime.datetime.now()

#     for l in range(len(c)):

#         psi.append(hash(tuple(D[l])))
#     #print(psi)
#     endtime = datetime.datetime.now()
     
#     tt1=endtime-starttime #jishi

#     Times1.append((tt0+tt1).microseconds/10000)

#     ###########################################################################
#     ###########################################################################

#     starttime =datetime.datetime.now()

#     for l in range(len(c)):

#         A3 = np.random.randint(0, 2, size=(len(c),len(c)))

#         e = np.random.randint(0, 2, size=len(c))

#         psi_=  np.dot(A3, np.array(v)) + e 

#         prg.append(psi_)
#     #print(psi)
#     endtime = datetime.datetime.now()
     
#     tt2=endtime-starttime #jishi

#     Times2.append((tt0+tt2).microseconds/10000)
    

fig = plt.figure()
an = plt.subplot(111) 

plt.xlabel('n')
plt.ylabel('CPU Time')
an.plot(Times1, label='CM20')
an.plot(Times2, label='Ours')

plt.legend()

#plt.plot(k0)
plt.show()




