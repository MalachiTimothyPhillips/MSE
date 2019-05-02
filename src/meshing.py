import numpy as np
def make_scatter(p):
    """
    Explicitly form the Q matrix for the scatter operation.
    Gather is simply Q.T.
    """
    nglobal = 2*(p+1)**2 - (p+1)
    nlocal = 2*(p+1)**2
    ndof_elem = (p+1)**2
    Q = np.zeros((nlocal,nglobal))
    local_id_count = 0
    global_id_count = 0
    for i in range(ndof_elem):
        Q[local_id_count,global_id_count] = 1
        local_id_count += 1
        global_id_count += 1
    for i in range(p+1):
        for j in range(p+1):
            if j == 0:
                Q[local_id_count,i*(p+1)+p] = 1
            else:
                Q[local_id_count,global_id_count] = 1
                global_id_count += 1
            local_id_count += 1
    return Q
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
    Q = make_scatter(2)
    import numpy.linalg as la
    assert np.allclose(Q_gold,Q)
    Q = make_scatter(1)
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
    Q = make_scatter(p)
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
