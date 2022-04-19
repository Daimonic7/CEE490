# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:14:33 2022

@author: Daimonic
"""

import numpy as np
from sympy import *
#A matrix
Afile = open(r"C:\Users\dainf\Downloads\A.dat", 'r')
A = []
for row in Afile:
    A.append([float(x) for x in row.split()])
#print(type(A[1][1]))

#b matrix
bfile = open(r"C:\Users\dainf\Downloads\b.dat", 'r')
b = []
for row in bfile:
    b.append([float(x) for x in row.split()])
#print(type(b[1]))

#Check
CM = np.concatenate((A, b), axis=1)
answer = Matrix(CM).rref()
print("The built-in python answer is:")
print(answer)
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
print("My solution is:")
print(X)
print("")

#Report time
import time
start = time.time()
a = 0
for i in range(1000):
    a = a + (i**100)
end = time.time()
print("The time of execution of above program is :", end-start)





