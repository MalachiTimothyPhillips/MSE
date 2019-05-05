import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from meshing import *
import os

# ### SG setup

exec(open('semsetup.py').read())
debug=True
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

def uinit(x1,x2,y1,y2):
    Lx = 4
    y_comp1 = 2/3*np.pi*y1*y1+1/2*np.pi
    y_comp2 = 2/3*np.pi*y2*y2+1/2*np.pi
    cx1 = np.multiply(np.cos(np.pi*x1/Lx),np.cos(np.pi*y1)) # Better choice of C
    cx2 = np.multiply(np.cos(np.pi*x2/Lx),np.cos(np.pi*y2)) # Better choice of C
    cy1 = np.multiply(np.sin(np.pi*x1/Lx),np.sin(y_comp1))
    cy2 = np.multiply(np.sin(np.pi*x2/Lx),np.sin(y_comp2))
    u01 = np.exp(-(x1**2.+y1**2.))  # u0
    u02 = np.exp(-(x2**2.+y2**2.))  # u0
    src = 0.  # Source term
    return cx1,cx2, cy1,cy2, u01, u02, src
debug = True
def advdif(p1,p2,nu,T,nt,nplt):
    """
    Problem simulates advection/diffusion equation in a two dimensional rectangular domain,
    where the x dimension goes in -2 to 2
    and the y dimension goes in -1 to 1
    """
    ah1,bh1,ch1,dh1,z1,w1 = semhat(p1)
    ah2,bh2,ch2,dh2,z2,w2 = semhat(p2)
    nx = (p1+1)+(p2+1)-1
    ny1 = p1+1
    nx1 = p1+1
    npt_elem1 = p1+1
    npt_elem2 = p2+1
    Q, boundary = make_scatter_mse(p1,p2) # scatter operator, used for assembly
    x1 = z1-1 # x points associated with lhs element
    x2 = z2+1 # x points associated with rhs element
    y1 = z1
    y2 = z2
    [X1,Y1] = np.meshgrid(x1,y1)
    [X2,Y2] = np.meshgrid(x2,y2)
    X1 = X1.T
    Y1 = Y1.T
    X2 = X2.T
    Y2 = Y2.T
    cx1, cx2, cy1,cy2, u1, u2,src = uinit(X1,X2,Y1,Y2)
    Ie1 = sp.sparse.eye(npt_elem1).tocsr()
    Ie2 = sp.sparse.eye(npt_elem2).tocsr()
    Bbar1 = spp.kron(bh1,bh1)
    Dx1 = spp.kron(Ie1,dh1)
    Dy1 = spp.kron(dh1,Ie1)
    Bbar2 = spp.kron(bh2,bh2)
    Dx2 = spp.kron(Ie2,dh2)
    Dy2 = spp.kron(dh2,Ie2)
    cx1 = cx1.reshape((npt_elem1*npt_elem1,))
    cy1 = cy1.reshape((npt_elem1*npt_elem1,))
    cx2 = cx2.reshape((npt_elem2*npt_elem2,))
    cy2 = cy2.reshape((npt_elem2*npt_elem2,))
    Abar1 = spp.kron(ah1,bh1) + spp.kron(bh1,ah1)
    Abar2 = spp.kron(ah2,bh2) + spp.kron(bh2,ah2)
    dt = T/float(nt)
    dti = 1./dt
    al, bt = bdfext_setup()
    ndt = int(nt/nplt)
    if(ndt==0):
        ndt = 1

    # actual R to be using here
    R = restrict_mse(p1,p2,boundary)
    u1v = u1.reshape((npt_elem1*npt_elem1,))
    u2v = u2.reshape((npt_elem2*npt_elem2,))
    uL = np.hstack([u1v,u2v])
    u = R.T@R@Q.T@uL
    u1_plot,u2_plot = reorder_u_for_plot_mse(u,p1,p2)
    f1 = np.zeros((u.shape[0],))
    f2 = f1.copy()
    f3 = f2.copy()
    fb1 = f1.copy()
    fb2 = f1.copy()
    fb3 = f1.copy()

    t = 0.
    # Plot initial field
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    #surf = ax1.contourf(X,Y,uplot)
    surf = ax1.contourf(X1,Y1,u1_plot.T, levels=np.linspace(0.0,1,9))
    surf2 = ax1.contourf(X2,Y2,u2_plot.T, levels=np.linspace(0.0,1,9))
    fig.colorbar(surf)
    ax1.quiver(X1,Y1,cx1,cy1,scale=10,headwidth=5,headlength=10)
    ax1.quiver(X2,Y2,cx2,cy2,scale=10,headwidth=5,headlength=10)
    ax1.set_title('t=%f'%t)
    ax = fig.add_subplot(1,2,2,projection='3d')
    wframe = ax.plot_wireframe(X1, Y1, u1_plot)
    wframe = ax.plot_wireframe(X2, Y2, u2_plot)
    ax.set_xlabel('X')
    ax.set_zlim(0,1)
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    #plt.show()
    #exit()
    plot_counter = 0
    #fig.savefig(f"tmp_plot_{str(plot_counter).zfill(5)}.png")

    # Form local DOF matrices
    AL = spp.block_diag((Abar1,Abar2))
    BL = spp.block_diag((Bbar1,Bbar2))
    Be1 = Bbar1.copy()
    Be2 = Bbar2.copy()
    Abar = Q.T@AL@Q
    Bbar = Q.T@BL@Q

    for i in range(nt): # nt
        if(i<=2):
            ali = al[i,:]
            bti = bt[i,:]
        uL = Q@u
        u1 = uL[0:npt_elem1*npt_elem1]
        u2 = uL[npt_elem1*npt_elem1:]
        w1 = Be1@(cx1*(Dx1@u1)+cy1*(Dy1@u1))
        #if(i==0):
        #    print("=== DEBUG ===")
        #    print(f"u2 shape: {u2.shape}")
        #    print(f"Dx shape: {u2.shape}")

        w2 = Be2@(cx2*(Dx2@u2)+cy2*(Dy2@u2))
        w = np.hstack([w1,w2])
        w = Q.T@w
        ## Advection term, to be extrapolated
        f1 = -w
        ## Source term, to be extrapolated
        f1 += Bbar@(src+0.*u)
        ## Mass matrix, backward diff
        fb1 = Bbar@u
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
            plot_counter += 1
            plt.clf()
            fig1, ax1 = plt.subplots()
            u1_plot,u2_plot = reorder_u_for_plot_mse(u,p1,p2)
            #surf = ax1.contourf(X,Y,uplot)
            surf = plt.contourf(X1,Y1,u1_plot.T, levels=np.linspace(0.0,1,9))
            surf2 = plt.contourf(X2,Y2,u2_plot, levels=np.linspace(0.0,1,9))
            fig1.colorbar(surf)
            ax1.quiver(X1,Y1,cx1,cy1,scale=10,headwidth=5,headlength=10)
            ax1.quiver(X2,Y2,cx2,cy2,scale=10,headwidth=5,headlength=10)
            ax1.set_title('t=%f'%t)
            plt.show()
            exit()
            #plt.pause(0.05)
            fig1.savefig(f"tmp_plot_{str(plot_counter).zfill(5)}.png")
            print('t=%f, umin=%g, umax=%g'%(t,np.amin(u),np.amax(u)))
    succ = 0
    # image processing stuff
    #os.system("convert   -delay 10   -loop 0   tmp_plot_*.png   two_elem_conv_diff.gif")
    #os.system("rm tmp_plot_*.png")
    return succ

p    = 15
nu   = 1.e-1
T    = 3.
nt   = 500
nplt = 5
succ = advdif(p,20,nu,T,nt,nplt)

