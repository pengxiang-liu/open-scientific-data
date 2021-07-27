# --------------------------------------------------------------------------- #
# Copyright 2020, Southeast University, Liu Pengxiang
# 
# A branch-and-cut approach to optimal power flow model in distribution 
# system. The model is formulated as a mixed-integer bilinear 
# programming.
#
# min   3 * x0 + 1 * x1 + 5 * y0 + 2 * y1 + 3 * y2
# s.t.  3 * x0 + 1 * x1 + 2 * y0 = 12
#       2 * x0 - 1 * x1 + 2 * y0 + 2 * y1 >= 10
#       5 * x1 + 2 * y0 + 5 * y1 + 3 * y2 >= 15
#       2 * x1 + 3 * y2 >= 5
#       y0 * y1 = y2 (bilinear constraint)
#       0.5 <= y0 <= 8, 0.5 <= y1 <= 8, 0.5 <= y2 <= 8
#       x0 and x1 are binary; y0, y1 and y2 are continuous
#
# --------------------------------------------------------------------------- #


import math
import numpy as np
import gurobipy as gp


# This function defines the nonconvex MIP model
# --------------------------------------------------------------------------- #
# 
def non_convex_mip():
    
    # 1. formulate matrix
    c = np.array([3, 1])
    d = np.array([5, 2, 3])
    M = np.array([[ 3,  1,  2,  0,  0],
                  [-3, -1, -2,  0,  0],
                  [ 2, -1,  2,  1,  0],
                  [ 5,  0,  2,  5,  3],
                  [ 0,  2,  0,  0,  3]])
    A = M[:, 0:2]
    B = M[:, 2:5]
    rhs = np.array([12, -12, 10, 15, 5])
    lby = np.array([0.5, 0.5, 0.5])
    uby = np.array([8.0, 8.0, 8.0])
    bil = np.array([[0, 1, 2]])

    # 2. spatial branching
    spatial_branching(c, d, A, B, rhs, lby, uby, bil)
    print(sol)


# This function defines the spatial branching algorithm
# --------------------------------------------------------------------------- #
# 
def spatial_branching(c, d, A, B, rhs, lby, uby, bil):
    
    global sol, obj, sgn
    # 1. Solve the bilinear relaxation model
    ret = bilinear_relaxation(c, d, A, B, rhs, lby, uby, bil)
    
    # 2. Branching
    # 1) if the problem is infeasible
    if ret['sgn'] == -1:
        sgn = -1
    # 2) if the solution is worse than the incumbent
    if ret['sgn'] ==  1 and ret['obj'][0] >= obj:
        sgn = -1
    # 3) if the solution is better than the incumbent
    if ret['sgn'] ==  1 and ret['obj'][0] <  obj:
        x = ret['x']
        y = ret['y']
        # 3.1) branching procedure
        gap = 0
        br_info = {'index': -1, 'theta': 0, 'delta': -100}
        for i in range(np.shape(bil)[0]):
            ix, iy, iz = bil[i, 0], bil[i, 1], bil[i, 2]
            rho = np.abs(y[iz] - y[ix] * y[iy])
            if rho >= 1e-2:
                br_info = branch_score(y, lby, uby, ix, iy, iz, br_info)
                br_info = branch_score(y, lby, uby, iy, ix, iz, br_info)
                gap = gap + rho
            index = br_info['index']
            theta = br_info['theta']
            delta = br_info['delta']
            if theta <= lby[index] + 1e-2 or theta >= uby[index] - 1e-2:
                theta = 0.5 * (lby[index] + uby[index])
        # 3.2) gap is narrow
        if gap == 0:
            sol = [ret['x'], ret['y']]
            obj = ret['obj']
            sgn = ret['sgn']
        # 3.3) keep branching
        else:
            # The upper branch
            lby_temp = np.copy(lby)
            lby_temp[index] = theta
            spatial_branching(c, d, A, B, rhs, lby_temp, uby, bil)
            # The lower branch
            uby_temp = np.copy(uby)
            uby_temp[index] = theta
            spatial_branching(c, d, A, B, rhs, lby, uby_temp, bil)

        
# This function solves the bilinear relaxation
# --------------------------------------------------------------------------- #
# 
def bilinear_relaxation(c, d, A, B, rhs, lby, uby, bil):
    
    # 1. Build gurobi model
    model  = gp.Model()

    # 2. Add variables
    x = model.addMVar(shape = len(c), vtype = 'B')
    y = model.addMVar(shape = len(d), vtype = 'C')

    # 3. Add constraints
    # 1) linear constraints
    model.addConstr(A @ x + B @ y >= rhs)
    # 2) lower and upper bounds
    model.addConstr(y >= lby)
    model.addConstr(y <= uby)
    # 3) McCormick inequality for bilinear term (y[0] * y[1] == y[2])
    for i in range(np.shape(bil)[0]):
        l0, l1 = lby[bil[i, 0]], lby[bil[i, 1]]
        u0, u1 = uby[bil[i, 0]], uby[bil[i, 1]]
        z0, z1, z2 = y[bil[i, 0]], y[bil[i, 1]], y[bil[i, 2]]
        model.addConstr(z2 >= l1 * z0 + l0 * z1 - l0 * l1)
        model.addConstr(z2 >= u1 * z0 + u0 * z1 - u0 * u1)
        model.addConstr(z2 <= u1 * z0 + l0 * z1 - l0 * u1)
        model.addConstr(z2 <= l1 * z0 + u0 * z1 - u0 * l1)
    
    # 4. Add objective
    objective = c @ x + d @ y
    model.setObjective(objective, gp.GRB.MINIMIZE)

    # 5. Bilinear terms (for testing)
    # model.addConstr(y[0] @ y[1] == y[2])
    # model.params.NonConvex = 2

    # 6. Solve
    # 1) call the solver
    model.optimize()
    # 2) diagnostic
    if model.status == gp.GRB.Status.OPTIMAL:
        ret = {'x': x.x, 'y': y.x, 'obj': objective.getValue(), 'sgn': 1}
    else:
        ret = {'sgn': -1}
    return ret


# This function defines the lazy constraint callback approach
# --------------------------------------------------------------------------- #
# 
def branch_score(y, lby, uby, ix, iy, iz, info):
    
    # 1. calculate the error
    rho = y[iz] - y[ix] * y[iy]

    # 2. determine the branch score
    if rho <= -1e-2:
        d = -rho / (1 + y[iy] - lby[iy])
        if d > info['delta']:
            info['index'] = ix
            info['theta'] = y[ix] - d
            info['delta'] = d
        d = -rho / (1 + uby[iy] - y[iy])
        if d > info['delta']:
            info['index'] = ix
            info['theta'] = y[ix] + d
            info['delta'] = d
    if rho >=  1e-2:
        d =  rho / (1 + uby[iy] - y[iy])
        if d > info['delta']:
            info['index'] = ix
            info['theta'] = y[ix] - d
            info['delta'] = d
        d =  rho / (1 + y[iy] - lby[iy])
        if d > info['delta']:
            info['index'] = ix
            info['theta'] = y[ix] + d
            info['delta'] = d
    
    # 3. return
    return info


# Main function
# --------------------------------------------------------------------------- #
# 
if __name__ == "__main__":

    # 1. global variables
    global sol, obj, sgn
    sol = np.array([0, 0, 0, 0, 0])
    obj = 100
    sgn = 0

    # 2. build the non-convex bilinear model
    non_convex_mip()