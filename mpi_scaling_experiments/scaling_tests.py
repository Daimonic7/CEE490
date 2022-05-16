from mpi4py import MPI
import numpy as np
import sys

from cg_variants.hs_cg import hs_cg
from cg_variants.cg_cg import cg_cg
from cg_variants.gv_cg import gv_cg
from cg_variants.pr_cg import pr_cg
from cg_variants.pipe_pr_cg import pipe_pr_cg

"""
Run parallel variants on model problem and return timings

mpiexec -n 2 python scaling_tests.py <n> <max_iter> <trial_name>
mpiexec -n 4 python scaling_tests.py 8192 1000 1x4-1


n = integer size of model problem
max_iter = number of iterations
trial_name = identifier for save data
"""

comm = MPI.COMM_WORLD
size = comm.Get_size() #number of MPI processes
rank = comm.Get_rank() #btw 0, size-1; process id number

trial_name = sys.argv[3]
n = int(sys.argv[1])
assert n%size == 0, "n must be a multiple of the number of processes"

E = 200*10**9
I = 30000*10**(-8)
L = 3
w = 15000

#Boundary Conditions
d0 = 0
dN = 0
x = np.linspace(0,L,n)
dx = L/(n-1)
#X = x[0:n]
print("x shape",np.shape(x))

#Matrix
M = np.zeros((n, n))
M[0][0] = 1
M[1][1] = -2
M[1][2] = 1
for i in range(2,n-2):
    M[i][i] = -2
    M[i][i-1] = 1
    M[i][i+1] = 1
M[n-2][n-2] = -2
M[n-2][n-3] = 1
M[n-1][n-1] = 1

#RHS
RHS = w*dx**2/(2*E*I)*(L*x-x**2)
RHS[0] = d0
RHS[n-1] = dN

if rank == 0: # Master Process

    sendRHS = RHS.reshape(size, -1)
    print("shape of sendbuf", np.shape(sendRHS))

    # Split into sub-arrays along required axis
    arrs = np.split(M, size, axis=1)

    # Flatten the sub-arrays
    raveled = [np.ravel(arr) for arr in arrs]

    # Join them back up into a 1D array
    sendM = np.concatenate(raveled)
    print("shape of sendM", np.shape(sendM))
else:
    sendM = None
    sendRHS = None

comm.Barrier() #barrier synchronization
if rank == 0:
    print("trial name: {}".format(trial_name))
    print("start distributing to {} ranks".format(size))
A = np.empty((n, n//size), dtype='float')
print("shape of A", np.shape(A))
b = np.empty(n//size, dtype='float')
print("shape of b", np.shape(b))
comm.Scatterv(sendM, A, root=0)
comm.Scatterv(sendRHS, b, root=0)


comm.Barrier()
if rank == 0:
    print("done distributing")


variants = [hs_cg,cg_cg,gv_cg,pr_cg,pipe_pr_cg]


max_iter = int(sys.argv[2])

for variant in variants:
    comm.Barrier()
    sol,t = variant(comm,A,b,max_iter)

    sol_raw = None
    if rank == 0:
        sol_raw = np.empty([size, n//size], dtype='float')
    comm.Gatherv(sol, sol_raw, root=0)

    if rank==0:
        sol_raw = np.reshape(sol_raw,(n))
        error = np.linalg.norm(np.ones(n)/np.sqrt(n)-sol_raw)
        print("{} error: {}".format(variant.__name__,error))
        #analytical solution
        Exact = -(w*x*(L**3 - 2*L*x**2 + x**3))/(24*E*I) 
        print("Exact shape ",np.shape(Exact))
        print(np.allclose(sol_raw,Exact,1E-7))
        ## now save results
        res = {"error":error,"timings":t}
        print(res)
        np.save("./data/{}/{}_{}".format(n,variant.__name__,trial_name),res,allow_pickle=True)



