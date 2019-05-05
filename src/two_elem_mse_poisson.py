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
def poisson(p1,p2):
    """
    Problem simulates advection/diffusion equation in a two dimensional rectangular domain,
    where the x dimension goes in -2 to 2
    and the y dimension goes in -1 to 1
    """
    ah1,bh1,ch1,dh1,z1,w1 = semhat(p1)
    ah2,bh2,ch2,dh2,z2,w2 = semhat(p2)
    nx = (p1+1)+(p2+1)-1
    ny1 = p1+1
    ny2 = p2+1
    npt_elem1 = p1+1
    npt_elem2 = p2+1
    Q, boundary = make_scatter_mse(p1,p2) # scatter operator, used for assembly
    x = np.hstack((z1-1,z2[1:]+1))
    x1 = z1-1 # x points associated with lhs element
    x2 = z2+1 # x points associated with rhs element
    y1 = z1
    y2 = z2
    [X1,Y1]=np.meshgrid(x1,y1)
    [X2,Y2]=np.meshgrid(x2,y2)
    X1 = X1.T
    X2 = X2.T
    Y1 = Y1.T
    Y2 = Y2.T
    Ie1 = sp.sparse.eye(npt_elem1).tocsr()
    Ie2 = sp.sparse.eye(npt_elem2).tocsr()
    Abar1 = spp.kron(ah1,bh1) + spp.kron(bh1,ah1)
    Bbar1 = spp.kron(bh1,bh1)
    Abar2 = spp.kron(ah2,bh2) + spp.kron(bh2,ah2)
    Bbar2 = spp.kron(bh2,bh2)
    R = restrict_mse(p1,p2,boundary)
    # Form local DOF matrices
    AL = spp.block_diag((Abar1,Abar2))
    BL = spp.block_diag((Bbar1,Bbar2))
    if debug:
        print("==== MATRIX SHAPES ====")
        print(f"R shape: {R.shape}")
        print(f"Q shape: {Q.shape}")
        print(f"AL shape: {AL.shape}")
        print(f"BL shape: {BL.shape}")
    A = R@Q.T@AL@Q@R.T
    B = Q.T@BL@Q
    rhs1 = f_func(X1,Y1)
    rhs1 = rhs1.reshape((npt_elem1*npt_elem1,))
    rhs2 = f_func(X2,Y2)
    rhs2 = rhs2.reshape((npt_elem2*npt_elem2,))
    fL = np.hstack([rhs1,rhs2])
    f = R@Q.T@BL@fL
    u = sppla.cg(A,f)[0]
    u = R.T@u
    u1,u2 = reorder_u_for_plot_mse(u,p1,p2)
    analytical1 = ufunc(X1,Y1)
    analytical2 = ufunc(X2,Y2)
    if debug:
        print(f"err: {la.norm(analytical1-u1)}")
        print(f"err: {la.norm(analytical2-u2)}")
    err_omega1 = la.norm(analytical1-u1)
    err_omega2 = la.norm(analytical2-u2)
    return err_omega1, err_omega2 
orders_1 = [5,10,15,20,30]
orders_2 = [5,10,15,20,30]
errs_1 = []
errs_2 = np.zeros((5,5))
marker_styles=['.','v','*','D','s']
for i,order in enumerate(orders_1):
    for j, order2 in enumerate(orders_2):
        err1, err2 = poisson(order,order2)
        errs_2[i,j] = err2
plt.figure(figsize=(4,3))
for i, order in enumerate(orders_2):
    plt.loglog(orders_2,errs_2[i,:], marker=marker_styles[i],label="$\\Omega_2$ Error, $N_1=$" + f"{order}")
plt.title("$\\Omega_2$ Error and Polynomial Order")
plt.xlabel("$N_2$")
plt.ylabel("$||u-\\tilde{u}||_F$")
plt.legend(loc=3)
plt.savefig('../poster/errplot_pdf.pdf', bbox_inches='tight',pad_inches = 0)
