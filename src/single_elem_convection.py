import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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

def uinit(x,y,Lx):
    y_comp = 2/3*np.pi*y*y+1/2*np.pi
    cx = np.multiply(np.cos(np.pi*x/Lx),np.cos(np.pi*y)) # Better choice of C
    cy = np.multiply(np.sin(np.pi*x/Lx),np.sin(y_comp))
    u = np.exp(-(x**2.+y**2.))  # u0
    ud = 0.0  # u = ud on Dirichlet
    bctyp = 3 # DDDD
    src = 0.  # Source term
    return cx, cy, u, ud, bctyp, src

def advdif(p,nu,T,nt,nplt):
    """
    Problem simulates advection/diffusion equation in a two dimensional rectangular domain,
    where the x dimension goes in -2 to 2
    and the y dimension goes in -1 to 1
    """
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
    cx, cy, u0, ud, bctyp, src = uinit(X,Y,Lx)
    # BC
    I1 = sp.sparse.eye(n1).tocsr()
    if(bctyp==1): # DDNN
        Rx = I1[1:-1,:].toarray() # Dirichlet left and right, Neumann top/bottom
        Ry = I1.toarray()
    elif(bctyp==2): # DNNN
        Rx = I1[1:,:].toarray() # Dirichlet left, Neumann everywhere
        Ry = I1.toarray()
    elif(bctyp==3): # DDDD
        Rx = I1[1:-1,:].toarray()
        Ry = I1[1:-1,:].toarray()
    elif(bctyp==4): # DNDN
        Rx = I1[1:,:].toarray()
        Ry = I1[1:,:].toarray()
    Ryt = Ry.T

    R = spp.kron(Ry,Rx)

    # FastDiagM setup
    Abar = Lx/Ly*spp.kron(ah,bh) + Ly/Lx*spp.kron(bh,ah)
    Bbar = spp.kron(bh,bh)
    Bbar *= Lx*Ly*0.25

    Dx = spp.kron(I1, Lx/2*dh)
    Dy = spp.kron(Ly/2*dh,I1)

    Dxc = np.zeros((n1*n1,n1*n1))
    Dyc = np.zeros((n1*n1,n1*n1))

    D_full = Dx.toarray()
    Dy_full = Dy.toarray()

    cx = cx.reshape((n1*n1,))
    cy = cy.reshape((n1*n1,))

    # Somehow this is the correct thing to do here...
    for i in range(n1*n1):
        Dxc[i] = cy[i]*D_full[i]
        Dyc[i] = cx[i]*Dy_full[i]
    
    Cbar = Bbar@Dxc+Bbar@Dyc
    Cbar = spp.csr_matrix(Cbar)
    cx = cx.reshape((n1,n1))
    cy = cy.reshape((n1,n1))


    dt = T/float(nt)
    dti = 1./dt
    al, bt = bdfext_setup()
    ndt = int(nt/nplt)
    if(ndt==0):
        ndt = 1
    ons = ud * np.ones(u0.shape)
    ub = ons - Rx.T.dot(Rx.dot(ons).dot(Ryt)).dot(Ry) # u=ud on D bndry
    u = Rx.T.dot(Rx.dot(u0).dot(Ry.T)).dot(Ry) + ub
    #f1 = np.zeros(Rx.dot(u0.reshape((n1,n1))).dot(Ryt).shape)
    f1 = np.zeros((n1*n1,))
    f2 = f1.copy()
    f3 = f2.copy()
    fb1 = f1.copy()
    fb2 = f1.copy()
    fb3 = f1.copy()

    t = 0.
    # Plot initial field
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    #surf = ax1.contourf(X,Y,u, levels=[0.1,0.25,0.5,1])
    surf = ax1.contourf(X,Y,u, levels=np.linspace(0.0,1,9))
    fig.colorbar(surf)
    ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
    ax1.set_title('t=%f'%t)
    ax1.set_autoscale_on(False)
    ax = fig.add_subplot(1,2,2,projection='3d')
    wframe = ax.plot_wireframe(X, Y, u)
    ax.set_xlabel('X')
    ax.set_zlim(0,1)
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    plot_counter = 0
    fig.savefig(f"tmp_plot_{str(plot_counter).zfill(5)}.png")
    u = u.reshape((n1*n1,))
    ub = ub.reshape((n1*n1,))

    for i in range(nt): # nt
        if(i<=2):
            ali = al[i,:]
            bti = bt[i,:]
        ## Advection term, to be extrapolated
        f1 = -1.0*Cbar@u
        ## Source term, to be extrapolated
        f1 += Bbar@(src+0.*u)
        ## Mass matrix, backward diff
        fb1 = Bbar@u
        ## RHS, Everything
        f = - (dti)*( bti[1]*fb1 + bti[2]*fb2 + bti[3]*fb3 )\
                + ali[0]*f1 + ali[1]*f2 + ali[2]*f3\
                - nu * Abar@ub
        # Save old states
        fb3 = fb2.copy()
        fb2 = fb1.copy()
        f3 = f2.copy()
        f2 = f1.copy()
        # Set up FDM solve
        h1 = nu
        h2 = bti[0] * dti
        Hbar = h2 * Bbar + nu*Abar
        H = R@Hbar@R.T
        RHS = R@f
        ug = sppla.cg(H,RHS)[0]
        u = R.T@ug+ub
        #u = u.reshape(int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0])))
        t  = float(i+1)*dt
        
        if((i+1)%ndt==0 or i==nt-1):
            u = u.reshape((n1,n1))
            plot_counter += 1
            plt.clf()
            ax1 = fig.add_subplot(1,2,1)
            surf = ax1.contourf(X,Y,u, levels=np.linspace(0.0,1,9))
            #surf = ax1.contourf(X,Y,u)
            fig.colorbar(surf)
            ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
            ax1.set_title('t=%f'%t)
            ax = fig.add_subplot(1,2,2,projection='3d')
            wframe = ax.plot_wireframe(X, Y, u)
            ax.set_zlim(0,1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('u')
            #plt.pause(0.05)
            fig.savefig(f"tmp_plot_{str(plot_counter).zfill(5)}.png")
            print('t=%f, umin=%g, umax=%g'%(t,np.amin(u),np.amax(u)))
            u = u.reshape((n1*n1,))

    succ = 0
    # image processing stuff
    os.system("convert   -delay 10   -loop 0   tmp_plot_*.png   single_elem_conv_diff.gif")
    os.system("rm tmp_plot_*.png")
    return succ

p    = 30
nu   = 1.e-2
T    = 10.
nt   = 2000
nplt = 100
succ = advdif(p,nu,T,nt,nplt)

