import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import os
# ### SG setup
exec(open('semsetup.py').read())
import scipy.linalg as sl
import time
from mpl_toolkits.mplot3d import axes3d
def f_func(x,y):
    return 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
def ufunc(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
def poisson(p):
    ah,bh,ch,dh,z,w = semhat(p)
    n1 = p + 1
    a = -2
    b = 2
    Lx = b-a
    Ly = 2
    x = (b-a)*0.5*(z+1)+a
    y = z
    [X,Y] = np.meshgrid(x,y)
    X = X.T
    Y = Y.T
    # BC
    I1 = sp.sparse.eye(n1).tocsr()
    Rx = I1[1:-1,:].toarray()
    Ry = I1[1:-1,:].toarray()
    Ryt = Ry.T
    R = spp.kron(Ry,Rx)
    bhx = Lx/2*bh
    bhy = Ly/2*bh
    dhx = Lx/2*dh
    dhy = Ly/2*dh
    ahx = 2/Lx*ah
    ahy = 2/Ly*ah
    Abar = spp.kron(bhx,ahy)+spp.kron(ahx,bhy)
    Bbar = Lx*Ly*0.25*spp.kron(bh,bh)
    rhs = f_func(X,Y)
    rhs = rhs.reshape((n1*n1,))
    analytical = ufunc(X,Y)
    A = R@Abar@R.T
    rhs = R@Bbar@rhs
    u = sppla.cg(A,rhs)[0]
    u = R.T@u
    u = u.reshape((n1,n1))
    # Plot initial field
    #fig = plt.figure(figsize=(12,6))
    #ax1 = fig.add_subplot(1,2,1)
    #surf = ax1.contourf(X,Y,u)
    #fig.colorbar(surf)
    #ax = fig.add_subplot(1,2,2,projection='3d')
    #wframe = ax.plot_wireframe(X, Y, u)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('u')
    #plt.show()

    #fig = plt.figure(figsize=(12,6))
    #ax1 = fig.add_subplot(1,2,1)
    #surf = ax1.contourf(X,Y,analytical)
    #fig.colorbar(surf)
    #ax = fig.add_subplot(1,2,2,projection='3d')
    #wframe = ax.plot_wireframe(X, Y, analytical)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('u')
    #plt.show()
    #print(f"u:\n{np.around(u,decimals=2)}")
    #print(f"analytical:\n{np.around(analytical, decimals=2)}")
    #print(f"err:\n{np.around(u-analytical, decimals=2)}")
    #print(f"err:\n{la.norm(u-analytical)}")
    return la.norm(u-analytical)
orders = [5,10,20,25,40,50]
errs = []
for order in orders:
    err = poisson(order)
    errs.append(err)
plt.loglog(orders,errs)
plt.xlabel("$p$, polynomial order")
plt.ylabel("$||u-\\tilde{u}||_F$")
plt.show()
