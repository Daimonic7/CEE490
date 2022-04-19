# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:58:16 2022

@author: Daimonic
"""
#Optimal Omega for SOR

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig

A = np.array([[1, -0.9], [-0.9, 1]])
b = np.array([0.6, 0.7])

N = len(A)
#Diagonal 
D = np.zeros((N,N))
for i in range(0,N):
    D[i][i] = A[i][i]
#Upper
U = np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        if i < j:
            U[i][j] = A[i][j]          
#Lower
L = np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        if i > j:
            L[i][j] = A[i][j] 
            
S = np.array([[8,0],[1,9]]) 
T = np.array([[0,1],[0,0]])
def SR(S,T):
    invS = inv(S)
    matrix = np.matmul(invS,T)
    value,vector = eig(matrix)
    sr = max(abs(value))
    return sr

#print(max(abs(SR(S,T))))
      
w = np.linspace(1,2,100)
sr = np.zeros(len(w))
for i in range(len(w)):
    S = D + w[i]*L
    T = (1-w[i])*D - w[i]*U
    sr[i] = SR(S,T)
    
x = np.where(sr == np.amin(sr))
x = x[0][0]          
print('Optimal omega: {}'.format(w[x]))            
            
            
            
            