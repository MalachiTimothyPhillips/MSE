import numpy as np
def make_scatter(p):
    """
    Explicitly form the Q matrix for the scatter operation.
    Gather is simply Q.T.
    Also, determine the global ids that correspond to the boundary of the system
    """
    nglobal = 2*(p+1)**2 - (p+1)
    nlocal = 2*(p+1)**2
    ndof_elem = (p+1)**2
    Q = np.zeros((nlocal,nglobal))
    local_id_count = 0
    global_id_count = 0
    boundary_nodes=set()
    for i in range(p+1):
        for j in range(p+1):
            Q[local_id_count, global_id_count] = 1
            local_id_count += 1
            if i == 0 or i == p or j == 0:
                boundary_nodes.add(global_id_count)
            global_id_count += 1
    for i in range(p+1):
        for j in range(p+1):
            if j == 0:
                Q[local_id_count,i*(p+1)+p] = 1
            else:
                Q[local_id_count,global_id_count] = 1
                if i == 0 or i == p or j == p:
                    boundary_nodes.add(global_id_count)
                global_id_count += 1
            local_id_count += 1
    return Q, boundary_nodes 
def mask(p):
    nglobal = 2*(p+1)**2 - (p+1)
    nelem = (p+1)
    bools = np.ones(nglobal)
    global_id_count = 0
    # E1 subsetting
    for i in range(nelem):
        for j in range(nelem):
            global_id = i*(p+1)+j
            global_id_count += 1
            if i == 0 or j == 0 or i == p:
                bools[global_id]=0
    # E2 subsettings
    for i in range(nelem):
        for j in range(nelem):
            #global_id = (p+1)**2 + i*p + j
            if i == 0 or j == p or i == p:
                bools[global_id_count]=0
            if j != 0:
                global_id_count += 1
    return bools
def restrict(p, boundary):
    num_global_ids = 2*(p+1)**2 - (p+1)
    # rectangle is made up of [2*(p+1)-1]x[p+1] nodes
    xlength = 2*(p+1)-1
    ylength = (p+1)
    perimeter = 2*xlength + 2*ylength - 4 # don't double count the corners
    num_real_dofs = num_global_ids - perimeter
    R = np.zeros((num_real_dofs,num_global_ids))
    real_dof_count = 0
    for dof in range(num_global_ids):
        if dof not in boundary:
            R[real_dof_count,dof] = 1
            real_dof_count += 1
    return R
def reorder_u(u,p):
    nx,ny = u.shape
    u_vec = np.zeros(nx*ny)
    entry = 0
    for j in range(p+1):
        for i in range(p+1):
            u_vec[entry] = u[i,j]
            entry += 1
    for j in range(p+1):
        for i in range(p+1,nx):
            u_vec[entry] = u[i,j]
            entry += 1
    return u_vec
def reorder_u_for_plot(u,p):
    npoints = u.shape[0]
    nx = int(npoints/(p+1))
    umat = np.zeros((2*(p+1)-1,(p+1)))
    entry = 0
    for j in range(p+1):
        for i in range(p+1):
            umat[i,j] = u[entry]
            entry += 1
    for j in range(p+1):
        for i in range(p+1,nx):
            umat[i,j] = u[entry]
            entry += 1
    return umat
if __name__ == "__main__":
    Q_gold = np.zeros((18,15))
    for i in range(9):
        Q_gold[i,i] = 1
    Q_gold[9,2]=1
    Q_gold[12,5] = 1
    Q_gold[15,8]=1
    Q_gold[10,9]=1
    Q_gold[11,10]=1
    Q_gold[13,11]=1
    Q_gold[14,12] = 1
    Q_gold[16,13] = 1
    Q_gold[17,14] = 1
    Q,_ = make_scatter(2)
    import numpy.linalg as la
    assert np.allclose(Q_gold,Q)
    Q,_ = make_scatter(1)
    Q_gold = np.zeros((8,6))
    for i in range(4):
        Q_gold[i,i] = 1
    Q_gold[4,1] = 1
    Q_gold[6,3] = 1
    Q_gold[5,4] = 1
    Q_gold[7,5] = 1
    assert np.allclose(Q_gold,Q)

    # Now suppose we take two high order elements
    p = 30
    nglobal = 2*(p+1)**2-(p+1)
    u = np.arange(0,nglobal) # real u
    Q, _ = make_scatter(p)
    uL = Q@u # uL should have local degrees
    nlocal = uL.shape[0]
    nelem1 = (p+1)**2
    local_id = 0
    global_id = 0
    for i in range(nelem1):
        assert np.isclose(uL[i],i) # correct so far
        local_id += 1
        global_id += 1
    for i in range(p+1):
        for j in range(p+1):
            if(j == 0):
                assert np.isclose(uL[local_id],u[i*(p+1)+p])
            else:
                np.isclose(uL[local_id],u[global_id])
                global_id += 1
            local_id += 1
    # Restriction test
    p = 2
    Q,boundary = make_scatter(p)
    R = restrict(p,boundary)

    gold_boundary=set([0,1,2,3,6,7,8,9,10,12,13,14])
    assert gold_boundary == boundary
    uglob = np.arange(15)
    print(f"u_global: {uglob}")
    u = R@uglob
    print(f"u restricted:{u}")
    print(f"R.T@u :{R.T@u}")

    p = 3
    Q,boundary = make_scatter(p)
    R = restrict(p,boundary)

    gold_boundary=set([0,1,2,3,4,8,12,13,14,15,16,17,18,21,24,25,26,27])
    assert gold_boundary == boundary
    uglob = np.arange(28)
    print(f"u_global: {uglob}")
    u = R@uglob
    print(f"u restricted:{u}")
    print(f"R.T@u :{R.T@u}")
