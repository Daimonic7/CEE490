# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:24:50 2022

@author: Daimonic
"""

from scipy.linalg import solve
import numpy as np

def SOR(A, b, x, w):
  A = np.array(A); b = np.array(b); x = np.array(x);
  n = len(A)
  xnew = np.zeros(n)
  for i in range(n):
    xnew[i] = (1 - w)*x[i] + w*(b[i] - sum([A[i, j]*x[j] for j in range(i + 1, n)]) - sum([A[i, j]*xnew[j] for j in range(i)]))/A[i, i]
  return xnew

A = [[1, -0.9], [-0.9, 1]]
b = [0.6, 0.7]
x = [[1,1]]

#A = [[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]]
#b = [7.85, -19.3, 71.4]
#x = [[1, 1, 1.]]

MaxIter = 100
ErrorTable = [1]
eps = 0.0001
i = 1

omega = 1.4

while i <= MaxIter and abs(ErrorTable[i - 1]) > eps: 
  xi = SOR(A, b, x[i - 1],omega)
  x.append(xi)
  ei = np.linalg.norm(x[i] - x[i - 1])/np.linalg.norm(x[i]) 
  ErrorTable.append(ei) 
  i+=1
  
print("x:",np.array(x))
print("ErrorTable:",np.vstack(ErrorTable))




