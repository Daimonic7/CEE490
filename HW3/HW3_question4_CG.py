# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:36:20 2022

@author: Daimonic
"""
import sys
import numpy as np

#example A and b
A = np.array([[ 0.7444, -0.5055, -0.0851],
              [-0.5055,  3.4858,  0.0572],
              [-0.0851,  0.0572,  0.4738]])
b = np.array([-0.0043,  2.2501,  0.2798])
x0 = np.array([0, 0, 0])

def LinearCG(A, b, x0, tol=1e-5):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)
    
    num_iter = 0
    curve_x = [xk]
    PK = []
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        PK.append(pk)
        num_iter += 1
        
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xk, rk_norm))
    
    print('\nSolution: \t x = {}'.format(xk))
    
    print('\nSearch directions: \t {}\n'.format(PK))
    
    a = curve_x[1] - curve_x[2]
    b = PK[0]
    print(np.dot(np.dot(a,A),b))
    print(PK)
    print(a.dot(A))
    return np.array(curve_x)

print(LinearCG(A, b, x0))








