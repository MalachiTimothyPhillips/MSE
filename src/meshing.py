import numpy as np
# Hard coded for E=2 case needed for the presentation
def make_scatter(p):
    """
    Explicitly form the Q matrix for the scatter operation.
    With this, Q.T forms a scatter opeartion, and
    the FEM assembly is done simply as 
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
# TODO, may want to verify p=3 case, too.
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
    print(f"Error: {la.norm(Q_gold-Q)}")
    Q = make_scatter(1)
    Q_gold = np.zeros((8,6))
    for i in range(4):
        Q_gold[i,i] = 1
    Q_gold[4,1] = 1
    Q_gold[6,3] = 1
    Q_gold[5,4] = 1
    Q_gold[7,5] = 1
    print(f"Error: {la.norm(Q_gold-Q)}")

