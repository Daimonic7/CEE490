# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:25:30 2022

@author: Daimonic
"""

import numpy as np
from sympy import *
import time
start_time = time.time()

N = 100
A = np.random.uniform(low=0.0, high=10.0, size=(N,N))
print(A)

b = np.random.uniform(low=0.0, high=10.0, size=(N,1))
print(b)

#Check
CM = np.concatenate((A, b), axis=1)
answer = Matrix(CM).rref()
print("The built-in python answer is:")
#print(answer)
print("")

#Gauss elimination
N = len(A)
for k in range(0, N-1):
    for i in range(k+1, N):
        factor = A[i][k]/A[k][k]
        for j in range(k, N):
            A[i][j] = A[i][j] - factor*A[k][j]
        b[i][0] = b[i][0] - factor*b[k][0]
#print(A)

#Back substitution
X = [0] * N
X[N-1] = b[N-1][0]/A[N-1][N-1]
for k in range(N-2, -1, -1):
    sum = 0
    for j in range(k+1, N):
        sum = sum + A[k][j]*X[j]
    X[k] = (b[k][0] - sum)/A[k][k]
#print("My solution is:")
#print(X)
#print("")

#Report time
print("--- %s seconds ---" % (time.time() - start_time))





