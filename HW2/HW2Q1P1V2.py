# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 19:23:44 2022

@author: Daimonic
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.csr import csr_matrix
from scipy.sparse.extract import tril, triu
from scipy.sparse.construct import diags
Inf = np.inf

def getMatrix(n=10, isDiagDom=True):
    '''
    Return an nxn matrix which is the discretization matrix
    from finite difference for a diffusion problem.
    To visualize A: use plt.spy(A.toarray())
    '''
    # Representation of sparse matrix and right-hand side
    assert n >= 2
    n -= 1
    diagonal  = np.zeros(n+1)
    lower     = np.zeros(n)
    upper     = np.zeros(n)

    # Precompute sparse matrix
    if isDiagDom:
        diagonal[:] = 2 + 1/n**2
    else:
        diagonal[:] = 2
    lower[:] = -1  #1
    upper[:] = -1  #1
    # Insert boundary conditions
    # diagonal[0] = 1
    # upper[0] = 0
    # diagonal[n] = 1
    # lower[-1] = 0

    
    A = diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(n+1, n+1),
        format='csr')

    return A

def getAuxMatrix(A):
    '''
    return D, L, U matrices for Jacobi or Gauss-Seidel
    D: array
    L, U: matrices
    '''
    # m = A.shape[0]
    D = csr_matrix.diagonal(A)
    L = -tril(A, k=-1)
    U = -triu(A, k=1)
    return D, L, U

A = getMatrix(n=5)
x_true = np.random.randn(5)
b = A*x_true


def jacobiIternation(A, b, x0=None, numIter = 100):
    D,L,U = getAuxMatrix(A)
    nDim = A.shape[0]
    x = np.zeros((numIter+1, nDim))
    error = np.zeros(numIter)
    for k in range(numIter):
        x[k+1] = ((L+U)*x[k])/D + b/D
        error[k] = (norm(x[k+1] - x_true)/norm(x[k] - x_true))
        
    return x,error

def GaussSeidelIternation(A, b, x0=None, numIter = 100):
    D,L,U = getAuxMatrix(A)
    nDim = A.shape[0]
    x = np.zeros((numIter+1, nDim))
    error = np.zeros(numIter)
    for k in range(numIter):
        x[k+1] = ((U)*x[k])/(D-L) + b/(D-L)
        error[k] = (norm(x[k+1] - x_true)/norm(x[k] - x_true))
        
    return x,error








