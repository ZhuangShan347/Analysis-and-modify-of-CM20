
import numpy as np


def G(A, x,q):    
    v = []
    for i in range(2*len(x)):
        x = np.dot(A,x)%q      
        v.append(x[0])
        x = x[1:]
    return v


def prf(A,x,k,q):
    v = G(A, x, q)
    if len(k)!=len(x):
        print('key error!')
        #print(len(k),len(x))

    for i in range(len(k)):    
        v_ = []
        if k[0]==0:
            for j in range(0,len(x)):
     
                v_.append(v[j])
        else:
            for j in range(len(x),2*len(x)):

                v_.append(v[j])
        v = G(A, v_, q)
    return v_









#print(v)




