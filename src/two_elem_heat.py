import numpy as np
import scipy as sp
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from meshing import *
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
debug = False
def f_func(x,y,t):
    phi = np.sin(np.pi*x)*np.sin(np.pi*y)
    return np.exp(-t)*phi+2*np.pi*np.pi*phi-2*np.pi*np.pi*np.exp(-t)*phi
def ufunc(x,y,t):
    return (1-np.exp(-t))*np.sin(np.pi*x)*np.sin(np.pi*y)
def uinit(x,y):
    u = np.exp(-(x**2.+y**2.))  # u0
    return u
def heat(p,T,nstep):
    """
    Problem simulates advection/diffusion equation in a two dimensional rectangular domain,
    where the x dimension goes in -2 to 2
    and the y dimension goes in -1 to 1
    """
    ah,bh,ch,dh,z,w = semhat(p)
    nx = 2*(p+1)-1
    ny = p+1
    npt_elem = p+1
    Q, boundary = make_scatter(p) # scatter operator, used for assembly
    x = np.hstack((z-1,z[1:]+1))
    x1 = z-1 # x points associated with lhs element
    x2 = z+1 # x points associated with rhs element
    y = z
    [X,Y] = np.meshgrid(x,y)
    X = X.T
    Y = Y.T
    Ie = sp.sparse.eye(npt_elem).tocsr()
    Lx = 2
    Ly = 2 # all elements are merely the reference element
    Abar = Lx/Ly*spp.kron(ah,bh) + Ly/Lx*spp.kron(bh,ah)
    Bbar = spp.kron(bh,bh)
    Bbar *= Lx*Ly*0.25
    R = restrict(p,boundary)
    # Form local DOF matrices
    AL = spp.kron(spp.eye(2),Abar)
    BL = spp.kron(spp.eye(2),Bbar)
    if debug:
        print("==== MATRIX SHAPES ====")
        print(f"R shape: {R.shape}")
        print(f"Q shape: {Q.shape}")
        print(f"AL shape: {AL.shape}")
        print(f"BL shape: {BL.shape}")
    Abar = Q.T@AL@Q
    Bbar = Q.T@BL@Q
    dt = float(T/nstep)
    dti = float(1/dt)
    al,bt = bdfext_setup()
    u = reorder_u(uinit(X,Y),p)
    f1 = np.zeros(u.shape)
    f2 = np.zeros(u.shape)
    f3 = np.zeros(u.shape)
    for i in range(nstep):
        if(i<=2):
            ali=al[i,:]
            bti=bt[i,:]
        # Mass matrix, backward diff
        f1 = Bbar@u
        #RHS
        forcing = f_func(X,Y,float(i*dt))
        forcing = reorder_u(forcing,p)
        f = -(dti)*(bti[1]*f1+bti[2]*f2+bti[3]*f3) + Bbar@forcing
        f3 = f2.copy()
        f2 = f1.copy()
        Hbar = (bti[0] * dti * Bbar + Abar)
        H = R@Hbar@R.T
        rhs = R@f
        ug = sppla.cg(H,rhs)[0]
        u = R.T@ug
    u = reorder_u_for_plot(u,p)
    analytical = ufunc(X,Y,T)
    err = la.norm(analytical-u)
    print(f"Err: {err}")
    return err 
heat(15,10,1000)
