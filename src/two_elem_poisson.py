import numpy as np
import scipy as sp
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from meshing import *
# ### SG setup
exec(open('semsetup.py').read())
import scipy.linalg as sl
import time
from mpl_toolkits.mplot3d import axes3d
def f_func(x,y):
    return 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
def ufunc(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
debug = False
def poisson(p):
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
    A = R@Q.T@AL@Q@R.T
    B = Q.T@BL@Q
    rhs = f_func(X,Y)
    rhs = reorder_u(rhs,p)
    rhs = R@B@rhs
    u = sppla.cg(A,rhs)[0]
    u = R.T@u
    u = reorder_u_for_plot(u,p)
    analytical = ufunc(X,Y)
    if debug:
        print(f"err: {la.norm(analytical-u)}")
    err = la.norm(analytical-u)
    return err 
orders = [5,10,20]
errs = []
for order in orders:
    err = poisson(order)
    errs.append(err)
plt.loglog(orders,errs)
plt.xlabel("$p$, polynomial order")
plt.ylabel("$||u-\\tilde{u}||_F$")
plt.show()
