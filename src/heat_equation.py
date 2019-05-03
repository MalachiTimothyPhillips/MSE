import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import os
# ### SG setup
exec(open('semsetup.py').read())
def bdfext_setup():    # BDF/EXT3
    al = np.zeros((3,3))
    bt = np.zeros((3,4))
    al[0,0] = 1.
    al[1,0] = 2.
    al[1,1] = -1.
    al[2,0] = 3.
    al[2,1] = -3.
    al[2,2] = 1.
    bt[0,0] = 1.
    bt[0,1] = -1.
    bt[1,0] = 3./2.
    bt[1,1] = -2.
    bt[1,2] = 1./2.
    bt[2,0] = 11./6.
    bt[2,1] = -3.
    bt[2,2] = 3./2.
    bt[2,3] = -1./3.
    return al, bt
import scipy.linalg as sl
import time
from mpl_toolkits.mplot3d import axes3d
def f_func(x,y,t):
    phi = np.sin(np.pi*x)*np.sin(np.pi*y)
    return np.exp(-t)*phi+2*np.pi*np.pi*phi-2*np.pi*np.pi*np.exp(-t)*phi
def ufunc(x,y,t):
    return (1-np.exp(-t))*np.sin(np.pi*x)*np.sin(np.pi*y)
def uinit(x,y):
    u = np.exp(-(x**2.+y**2.))  # u0
    return u
def heat(p,T,nstep):
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
    u = uinit(X,Y)
    u = u.reshape((n1*n1,))
    A = R@Abar@R.T
    f1=np.zeros((n1*n1,))
    f2=f1.copy()
    f3=f1.copy()
    t = 0
    dt = float(T/nstep)
    dti = float(1/dt)
    al,bt = bdfext_setup()
    for i in range(nstep):
        if(i<=2):
            ali=al[i,:]
            bti=bt[i,:]
        # Mass matrix, backward diff
        f1 = Bbar@u
        # RHS
        forcing = f_func(X,Y,float(i*dt)).reshape((n1*n1,))
        f = -(dti)*(bti[1]*f1+bti[2]*f2+bti[3]*f3) + Bbar@forcing
        f3 = f2.copy()
        f2 = f1.copy()
        Hbar = (bti[0] * dti * Bbar + Abar)
        H = R@Hbar@R.T
        rhs = R@f
        ug = sppla.cg(H,rhs)[0]
        u = R.T@ug
    u = u.reshape((n1,n1))
    err = la.norm(ufunc(X,Y,T)-u)
    print(f"Error: {err}")
    return err
heat(15,10,1000)
