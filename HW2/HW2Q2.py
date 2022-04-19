# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:53:53 2022

@author: Daimonic
"""

import sys
import numpy as np
from sympy import *

def e(N):
    #Matrix A
    def tridiag(x, y, z, k1=-1, k2=0, k3=1):
        return np.diag(x, k1) + np.diag(y, k2) + np.diag(z, k3)
    x = [1]*(N+1)
    y = [-(2 + 4*(1/((N+1)**2)))]*(N+2)
    z = [1]*(N+1)
    z[0] = 0
    x[N] = 0
    y[0], y[N+1] = 1,1
    A = tridiag(x,y,z)
    A[0,0] = 1
    A[0,1] = 0
    A[N+1,N+1] = 1
    A[N+1,N+0] = 0
    
    print(A)
    print("")
    print(x,y,z)
    
    #Matrix b
    b = [0]*(N+2)
    b[N+1] = 10.
    print(b)
    #sys.exit()
    
    #check
    CM = np.column_stack( (A,b) )
    answer = Matrix(CM).rref()
    print(CM)
    print("The built-in python answer is:")
    print(answer)
    
    
    def TDMAsolver(ac, bc, cc, dc):
        nf = len(bc) # number of equations
        #ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy the array
        for i in range(1, nf-1):
            mc = ac[i]/bc[i-1]
            bc[i] = bc[i] - mc*cc[i-1]
            dc[i] = dc[i] - mc*dc[i-1]
    
        xc = [0]*(N+2)
        xc[-1] = dc[-1]/bc[-1]
    
        for l in range(nf-2, 0, -1):
            xc[l] = (dc[l]-cc[l]*xc[l+1])/bc[l]
            
        return xc
    
    print("")
    U = TDMAsolver(x,y,z,b)
    #print(U)
    
    #Evaluate Error
    def Ua(x):
        return (10*exp(2-2*x))*(exp(4*x)-1)/(exp(4)-1)
    
    X = np.linspace(0, 1, N+2)
    #print(X)
    sum = 0
    for i in range(N+2):
        sum = sum + (U[i] - Ua(X[i]))**2
        
    error = sqrt(sum)/(N+2)
    print("error =")
    print(float(error))
    
    return float(error)

import pandas as pd
import matplotlib.pyplot as plt

#create DataFrame
X = [10,20,40,80]
Y = []
for n in range(len(X)):
    Y.append(e(X[n]))
df = pd.DataFrame({'x': X, 'y': Y})

#create scatterplot
#plt.scatter(df.x, df.y)

import numpy as np

#perform log transformation on both x and y
xlog = np.log(df.x)
ylog = np.log(df.y)

#create log-log plot
plt.scatter(xlog, ylog)

#create log-log plot with labels
plt.scatter(xlog, ylog, color='purple')
plt.xlabel('Log(x)')
plt.ylabel('Log(y)')
plt.title('Log-Log Plot')



