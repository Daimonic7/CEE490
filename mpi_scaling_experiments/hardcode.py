import sys
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import time

start_time = time.time()

E = 200*10**9
I = 30000*10**(-8)
L = 3
N = 5
w = 15000

#Boundary Conditions
d0 = 0
dN = 0
x = np.linspace(0,L,N)
dx = L/(N-1)
x_int = x[1:N-1] #internal nodes

#Generate Matrix
N_mat = N-2
A = np.zeros((N_mat, N_mat))
A[0][0] = -2
A[0][1] = 1
for i in range(1,N_mat-1):
    A[i][i] = -2
    A[i][i-1] = 1
    A[i][i+1] = 1
A[N_mat-1][N_mat-1] = -2
A[N_mat-1][N_mat-2] = 1    

#def M(x):
#    return ((w*L*x)/2) - ((w*x**2)/2)

RHS = w*dx**2/(2*E*I)*(L*x_int-x_int**2)
di = np.zeros(N_mat)

def LinearCG(A, b, x0, tol=1e-14):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)
    
    num_iter = 0
    #curve_x = [xk]
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1
        #curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xk, rk_norm))
    
    print('\nSolution: \t x = {}'.format(xk))
        
    return xk

d_int = LinearCG(A, RHS, di)
d = np.insert(d_int,0,d0)
d = np.insert(d,N-1,dN)
print('solution with BC: {}'.format(d))

CM = np.column_stack((A,RHS))
answer = Matrix(CM).rref()
#print("The direct answer is:")
#print(answer)

#analytical solution
x2 = np.linspace(0,L,1000)
d2 = -(w*x2*(L**3 - 2*L*x2**2 + x2**3))/(24*E*I)    

plt.title("Deflection Graph (N = {})".format(N))
plt.plot(x, d, color="red")
plt.plot(x2, d2, color="blue")
plt.xlabel('Span (m)')
plt.ylabel('Deflection (m)')
plt.legend(["Numerical Sol.","Analytical Sol."])
plt.show()

#Report time
print("--- %s seconds ---" % (time.time() - start_time))




