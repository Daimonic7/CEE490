# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:13:29 2022

@author: Daimonic
"""

from sympy import *
import numpy as np 
import matplotlib.pyplot as plt

def IT(f, a, b, n):
  h = (b - a)/n
  It = 0
  for i in range(n):
      It += (h/2)*(f(a+h*i) + f(a+h*(i+1)))
  
  return It
  #return sum([(f(a + i*h) + f(a + (i + 1)*h))*h/2 for i in range(int(n))])

def f(x): 
    return 1/(sqrt(x+0.1)) + 1/((x-0.3)**2+0.5) - np.pi
#print(IT(f, 0, 1, 1))
#print(IT(f, 0, 1, 1000))

x = symbols('x')
A_sol = integrate(f(x), (x, 0, 1))
print('Analytical Solution: {}'.format(A_sol))

n = 100
x = np.arange(1,n+1)
y = np.zeros(n)
for n in x:
    y[n-1] = (A_sol - IT(f, 0, 1, n))/A_sol*100
   
xlog = np.log(x)
ylog = np.log(y)

print('Numerical Solution: {} with {} trapezoids'.format(IT(f,0,1,n),n))
print('Percent Error: {}%'.format((A_sol - IT(f, 0, 1, n))/A_sol*100))
#create log-log plot
plt.scatter(xlog, ylog)

#create log-log plot with labels
plt.scatter(xlog, ylog, color='purple')
plt.xlabel('Log(x)')
plt.ylabel('Log(y)')
plt.title('Log-Log Plot')
    
    
    
    
    
    
    
    
    
    
    
    