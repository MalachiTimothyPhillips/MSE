import numpy as np
import scipy as sp
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

def uinit(x,y,x1,x2,y1,y2):
    Lx = 2
    y_comp1 = 2/3*np.pi*y1*y1+1/2*np.pi
    y_comp2 = 2/3*np.pi*y2*y2+1/2*np.pi
    y_comp = 2/3*np.pi*y*y+1/2*np.pi
    #cx1 = np.multiply(np.cos(np.pi*x1/Lx),np.cos(np.pi*y1)) # Better choice of C
    #cx2 = np.multiply(np.cos(np.pi*x2/Lx),np.cos(np.pi*y2)) # Better choice of C
    #cy1 = np.multiply(np.sin(np.pi*x1/Lx),np.sin(y_comp1))
    #cy2 = np.multiply(np.sin(np.pi*x2/Lx),np.sin(y_comp2))
    #cx = np.multiply(np.cos(np.pi*x/Lx),np.cos(np.pi*y)) # Better choice of C
    #cy = np.multiply(np.sin(np.pi*x/Lx),np.sin(y_comp))
    cx1 = 0*x1
    cy1 = 0*x1
    cx2 = 0*x2
    cy2 = 0*x2
    cx = 0*x
    cy = 0*x
    u = np.exp(-(x**2.+y**2.))  # u0
    src = 0.  # Source term
    return cx, cy, cx1,cx2, cy1,cy2, u, src
debug = False
def advdif(p,nu,T,nt,nplt):
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
    [X1,Y1] = np.meshgrid(x1,y)
    [X2,Y2] = np.meshgrid(x2,y)
    X = X.T
    Y = Y.T
    X1 = X1.T
    Y1 = Y1.T
    X2 = X2.T
    Y2 = Y2.T
    cx, cy, cx1, cx2, cy1,cy2, u0,src = uinit(X,Y,X1,X2,Y1,Y2)
    Ie = sp.sparse.eye(npt_elem).tocsr()

    # FastDiagM setup
    Lx = 2
    Ly = 2 # all elements are merely the reference element
    Abar = Lx/Ly*spp.kron(ah,bh) + Ly/Lx*spp.kron(bh,ah)
    Bbar = spp.kron(bh,bh)
    Bbar *= Lx*Ly*0.25

    Dx = spp.kron(Ie, Lx/2*dh)
    Dy = spp.kron(Ly/2*dh,Ie)

    Dxc1 = np.zeros((npt_elem*npt_elem,npt_elem*npt_elem)) # based on an individual element
    Dyc1 = np.zeros((npt_elem*npt_elem,npt_elem*npt_elem))
    Dxc2 = np.zeros((npt_elem*npt_elem,npt_elem*npt_elem)) # based on an individual element
    Dyc2 = np.zeros((npt_elem*npt_elem,npt_elem*npt_elem))

    D_full = Dx.toarray()
    Dy_full = Dy.toarray()

    cx1 = cx1.reshape((npt_elem*npt_elem,))
    cy1 = cy1.reshape((npt_elem*npt_elem,))
    cx2 = cx2.reshape((npt_elem*npt_elem,))
    cy2 = cy2.reshape((npt_elem*npt_elem,))

    # Somehow this is the correct thing to do here...
    for i in range(npt_elem*npt_elem):
        Dxc1[i] = cy1[i]*D_full[i]
        Dyc1[i] = cx1[i]*Dy_full[i]
        Dxc2[i] = cy2[i]*D_full[i]
        Dyc2[i] = cx2[i]*Dy_full[i]
    cx1 = cx1.reshape((npt_elem,npt_elem))
    cy1 = cy1.reshape((npt_elem,npt_elem))
    cx2 = cx2.reshape((npt_elem,npt_elem))
    cy2 = cy2.reshape((npt_elem,npt_elem))
    
    Cbar1 = Bbar@Dxc1+Bbar@Dyc1
    Cbar1 = spp.csr_matrix(Cbar1)
    Cbar2 = Bbar@Dxc2+Bbar@Dyc2
    Cbar2 = spp.csr_matrix(Cbar2)

    dt = T/float(nt)
    dti = 1./dt
    al, bt = bdfext_setup()
    ndt = int(nt/nplt)
    if(ndt==0):
        ndt = 1
    #u = Rx.T.dot(Rx.dot(u0).dot(Ry.T)).dot(Ry)
    u  = u0
    f1 = np.zeros((nx*ny,))
    f2 = f1.copy()
    f3 = f2.copy()
    fb1 = f1.copy()
    fb2 = f1.copy()
    fb3 = f1.copy()

    # actual R to be using here
    R = restrict(p,boundary)
    u_vec = reorder_u(u,p)
    uinner=R@u_vec
    u = R.T@uinner
    uplot = reorder_u_for_plot(u,p)

    t = 0.
    # Plot initial field
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    surf = ax1.contourf(X,Y,uplot)
    fig.colorbar(surf)
    ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
    ax1.set_title('t=%f'%t)
    ax = fig.add_subplot(1,2,2,projection='3d')
    wframe = ax.plot_wireframe(X, Y, uplot)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    plt.pause(0.5)

    # Form local DOF matrices
    AL = spp.kron(spp.eye(2),Abar)
    BL = spp.kron(spp.eye(2),Bbar)
    CL = spp.kron(np.array([[1,0],[0,0]]), Cbar1)
    CL += spp.kron(np.array([[0,0],[0,1]]), Cbar2)

    for i in range(nt): # nt
        if(i<=2):
            ali = al[i,:]
            bti = bt[i,:]
        ## Advection term, to be extrapolated
        f1 = -1.0*Q.T@CL@Q@u
        ## Source term, to be extrapolated
        f1 += Q.T@BL@Q@(src+0.*u)
        ## Mass matrix, backward diff
        fb1 = Q.T@BL@Q@u
        ## RHS, Everything
        f = - (dti)*( bti[1]*fb1 + bti[2]*fb2 + bti[3]*fb3 )\
                + ali[0]*f1 + ali[1]*f2 + ali[2]*f3
        # Save old states
        fb3 = fb2.copy()
        fb2 = fb1.copy()
        f3 = f2.copy()
        f2 = f1.copy()
        # Set up FDM solve
        h1 = nu
        h2 = bti[0] * dti
        Hbar = Q.T@(h2 * BL+ h1*AL)@Q
        H = R@Hbar@R.T
        RHS = R@f
        ug = sppla.cg(H,RHS)[0]
        u = R.T@ug
        t  = float(i+1)*dt
        
        if((i+1)%ndt==0 or i==nt-1):
            uplot = reorder_u_for_plot(u,p)
            plt.clf()
            ax1 = fig.add_subplot(1,2,1)
            surf = ax1.contourf(X,Y,uplot)
            fig.colorbar(surf)
            ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
            ax1.set_title('t=%f'%t)
            ax = fig.add_subplot(1,2,2,projection='3d')
            wframe = ax.plot_wireframe(X, Y, uplot)
            u = u.reshape((nx*ny,))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('u')
            plt.pause(0.5)
            print('t=%f, umin=%g, umax=%g'%(t,np.amin(u),np.amax(u)))
    succ = 0
    return succ

p    = 15
nu   = 1.e-2
T    = 10.
nt   = 1000
nplt = 5
succ = advdif(p,nu,T,nt,nplt)

