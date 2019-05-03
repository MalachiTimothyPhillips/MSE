from meshing import *
p = 2
Q,boundary = make_scatter(p)
R = restrict(p,boundary)
assert set([0,1,2,3,6,7,8,9,10,12,13,14]) == boundary
# based on grid points
u_grid = np.zeros((5,3))
for i in range(5):
    for j in range(3):
        idx = j*(5)+i
        u_grid[i,j] = idx
u_vec = reorder_u(u_grid, p)
print(u_grid)
print(u_vec)
print(R@u_vec)
print(R.T@R@u_vec)
print(reorder_u_for_plot(R.T@R@u_vec,p))

