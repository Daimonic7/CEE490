# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:05:43 2022

@author: Daimonic
"""
import numpy as np
from sympy import *
import sys
import copy

x = [0.0,0.0]
eps = 1*10**(-10)

def F(x):
    f1 = (x[0]-2)**2 + (x[1]-3)**2 + (x[0]-2.1)*(x[1]-3.1) + 100*x[0] - 302.81
    f2 = 10*2.71828**(-1*x[0]) + 5*2.71828**(1-x[1]) - 100*x[1] + 399.2532
    return np.array([f1,f2])

def J(x):
    return np.array([[2*x[0] + x[1] + 92.9, x[0] + 2*x[1] - 8.1],
                    [-10*2.71828**(-x[0]), -5*2.71828**(1 - x[1]) - 100]])

#sys.exit()

def N(x, eps):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = F(x)
    F_norm = np.linalg.norm(F_value)  # l2 norm of vector
    iteration_counter = 0
    j = J(x).copy()
    while abs(F_norm) > eps and iteration_counter < 100:
        delta = np.linalg.solve(j, -F_value)
        #delta = (-(J(x).inv()*F_value))
        x = x + delta
        F_value = F(x)
        F_norm = np.linalg.norm(F_value)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    return x, iteration_counter, J(x), F_value

exact, numiter, A, b = N(x,eps)

print("Number of iterations: {}\nExact solution: {}".format(numiter, exact))

sys.exit()

#GS method
X = [[0,0]]
print(A,b)

def GS(A, b, x):
  A = np.array(A)
  n = len(A)
  xnew = np.zeros(n)
  for i in range(n):
      xnew[i] = (b[i] - sum([A[i, j]*x[j] for j in range(i+1, n)]) 
               - sum([A[i, j]*xnew[j] for j in range(i)]))/A[i, i]
  return xnew

MaxIter = 100
ErrorTable = [1]
eps = 0.0001
i = 1

while i <= MaxIter and abs(ErrorTable[i - 1]) > eps:
  xi = GS(A, b, X[i - 1])
  X.append(xi) 
  ei = np.linalg.norm(X[i] - X[i - 1])/np.linalg.norm(X[i]) 
  ErrorTable.append(ei)
  i+=1
print("x:",np.array(X))
print("ErrorTable:",np.vstack(ErrorTable))






