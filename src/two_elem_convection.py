import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as spp
import scipy.sparse.linalg as sppla


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

def uinit(x,y):
    cx = np.multiply(np.cos(np.pi*x),np.cos(np.pi*y)) + 0.5
    cy = np.multiply(np.sin(np.pi*x),np.sin(np.pi*y))
    u = np.exp(-(x+y)**2.)  # u0
    ud = 0.0  # u = ud on Dirichlet
    bctyp = 3 # DDDD
    src = 0.  # Source term
    return cx, cy, u, ud, bctyp, src

def advdif(p,nu,T,nt,nplt):
    ah,bh,ch,dh,z,w = semhat(p)
    n1 = p + 1
    x = z
    [X,Y] = np.meshgrid(x,x)
    X = X.T
    Y = Y.T
    cx, cy, u0, ud, bctyp, src = uinit(X,Y)
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
    Ax = Rx.dot(ah).dot(Rx.T)
    Bx = Rx.dot(bh).dot(Rx.T)
    Ay = Ry.dot(ah).dot(Ry.T)
    By = Ry.dot(bh).dot(Ry.T)
    Abar = spp.kron(ah,bh) + spp.kron(bh,ah)
    Bbar = spp.kron(bh,bh)
    wy,vy = sl.eigh(Ay,By) # Equation in direction y
    wx,vx = sl.eigh(Ax,Bx) # Equation in direction x
    vyt = vy.T
    vxt = vx.T
    ry = wy.shape[0]
    rx = wx.shape[0]
    sy = np.ones(ry)
    sx = np.ones(rx)
    
    Dh1 = wy + wx[:,np.newaxis]   # reshaped lambda x I + I x lambda
    Dh2 = sy + 0.*sx[:,np.newaxis] # reshaped I x I

    dt = T/float(nt)
    dti = 1./dt
    al, bt = bdfext_setup()
    ndt = int(nt/nplt)
    if(ndt==0):
        ndt = 1
    ons = ud * np.ones(u0.shape)
    ub = ons - Rx.T.dot(Rx.dot(ons).dot(Ryt)).dot(Ry) # u=ud on D bndry
    u = Rx.T.dot(Rx.dot(u0).dot(Ry.T)).dot(Ry) + ub
    f1 = np.zeros(Rx.dot(u0.reshape((n1,n1))).dot(Ryt).shape)
    f2 = f1.copy()
    f3 = f2.copy()
    fb1 = f1.copy()
    fb2 = f1.copy()
    fb3 = f1.copy()

    t = 0.
    # Plot initial field
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    surf = ax1.contourf(X,Y,u)
    fig.colorbar(surf)
    ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
    ax1.set_title('t=%f'%t)
    ax = fig.add_subplot(1,2,2,projection='3d')
    wframe = ax.plot_wireframe(X, Y, u)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    plt.pause(0.5)
    input("Press Enter to continue...")

    for i in range(nt): # nt
        if(i<=2):
            ali = al[i,:]
            bti = bt[i,:]

        # Advection term, to be extrapolated
        f1 = - Rx.dot(bh.dot(np.multiply(cx,dh.dot(u))\
                + np.multiply(cy,u.dot(dh.T))).dot(bh)).dot(Ryt)
        # Source term, to be extrapolated
        f1 = f1 + Rx.dot(bh.dot(src+0.*u).dot(bh)).dot(Ryt)
        # Mass matrix, backward diff
        fb1 = Rx.dot(bh.dot(u).dot(bh.T)).dot(Ryt)
        # RHS, Everything
        f = - (dti)*( bti[1]*fb1 + bti[2]*fb2 + bti[3]*fb3 )\
                + ali[0]*f1 + ali[1]*f2 + ali[2]*f3\
                - nu * Rx.dot(ah.dot(ub).dot(bh)+bh.dot(ub).dot(ah.T)).dot(Ryt)
        # Save old states
        fb3 = fb2.copy()
        fb2 = fb1.copy()
        f3 = f2.copy()
        f2 = f1.copy()
        # Set up FDM solve
        h1 = nu
        h2 = bti[0] * dti
        Dl = h1 * Dh1 + h2 * Dh2
        Hbar = h2 * Bbar + nu*Abar
        H = R@Hbar@R.T
        f = f.reshape(H.shape[0],1)
        ug = sppla.cg(H,f)[0]
        u = R.T@ug
        u = u.reshape(int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0]))) + ub
        t  = float(i+1)*dt
        
        if((i+1)%ndt==0 or i==nt-1):
            plt.clf()
            ax1 = fig.add_subplot(1,2,1)
            surf = ax1.contourf(X,Y,u)
            fig.colorbar(surf)
            ax1.quiver(X,Y,cx,cy,scale=10,headwidth=5,headlength=10)
            ax1.set_title('t=%f'%t)
            ax = fig.add_subplot(1,2,2,projection='3d')
            wframe = ax.plot_wireframe(X, Y, u)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('u')
            plt.pause(0.5)
            input("Press Enter to continue...")
            
            print('t=%f, umin=%g, umax=%g'%(t,np.amin(u),np.amax(u)))

    succ = 0
    return succ

p    = 30
nu   = 1.e-2
T    = 10.
nt   = 2000
nplt = 5
succ = advdif(p,nu,T,nt,nplt)

