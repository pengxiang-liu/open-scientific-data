# --------------------------------------------------------------------------- #
# Copyright 2020, Southeast University, Liu Pengxiang
# 
# A spatial branch-and-bound algorithm for mixed-integer bilinear programming,
# where each convexified branch is solved by cplex 12.10.
#
# min   3 * x0 + 1 * x1 + 5 * x2 + 2 * x3 + 3 * x4
# s.t.  3 * x0 + 1 * x1 + 2 * x2 == 12
#       2 * x0 - 1 * x1 + 2 * x2 + 2 * x3 >= 10
#       5 * x0 + 2 * x2 + 5 * x3 + 3 * x4 >= 15
#       2 * x1 + 3 * x4 >= 5
#       x2 * x3 = x4 (bilinear constraint)
#       0 <= x0 <= 1, 0 <= x1 <= 1
#       0.5 <= x2 <= 8, 0.5 <= x3 <= 8, 0.5 <= x4 <= 8
#       x0 and x1 are binary; x2, x3 and x4 are continuous
#
# min   c * x
# s.t.  A * x == d
#       B * x >= e
#       bilinear(x) == 0
# --------------------------------------------------------------------------- #


import os
import sys
import math
import cplex
import numpy as np

from scipy.sparse import coo_matrix


# generic callback
class my_callback(object):

    def __init__(self, lb, ub, bl):
        self.branches = 0
        self.lb = lb
        self.ub = ub
        self.bl = bl
    
    def invoke(self, context):
        try:
            if context.in_candidate():
                self.incumbent(context)
            if context.in_branching():
                self.spatial_branching(context)
        except:
            info = sys.exc_info()
            print('## Exception in callback: ', info[0])
            print('##                        ', info[1])
            print('##                        ', info[2])
            raise
    
    def incumbent(self, context):
        for row in self.bl:
            x = context.get_candidate_point(row[0])
            y = context.get_candidate_point(row[1])
            z = context.get_candidate_point(row[2])
            if np.abs(x * y - z) >= 1e-3:
                context.reject_candidate()

    def spatial_branching(self, context):
        status = context.get_relaxation_status()
        if status != context.solution_status.optimal:
            if status != context.solution_status.optimal_infeasible:
                return
        # spatial branching
        for row in self.bl:
            x = context.get_relaxation_point(row[0])
            y = context.get_relaxation_point(row[1])
            z = context.get_relaxation_point(row[2])
            if np.abs(x * y - z) >= 1e-3:
                obj = context.get_relaxation_objective()
                # up branch
                ix, iy, iz = row[0], row[1], row[2]
                lx, ux, ly, uy = x, self.ub[ix], self.lb[iy], self.ub[iy]
                lhs, rhs, senses = convex_hull(ix, iy, iz, lx, ux, ly, uy)
                up_child = context.make_branch(
                    obj, variables = [(ix, 'L', x)], 
                    constraints = [(lhs[i], senses[i], rhs[i]) for i in range(4)]
                )
                # down branch
                ix, iy, iz = row[0], row[1], row[2]
                lx, ux, ly, uy = self.lb[ix], x, self.lb[iy], self.ub[iy]
                lhs, rhs, senses = convex_hull(ix, iy, iz, lx, ux, ly, uy)
                up_child = context.make_branch(
                    obj, variables = [(ix, 'U', x)], 
                    constraints = [(lhs[i], senses[i], rhs[i]) for i in range(4)]
                )


# This function sets the parameters
# --------------------------------------------------------------------------- #
# 
def prob_settings(prob):

    # turn off presolve
    prob.parameters.threads.set(1)
    prob.parameters.preprocessing.reduce.set(0)
    prob.parameters.preprocessing.linear.set(0)
    # prob.parameters.preprocessing.presolve.set(0)
    
    # return
    return prob


# add the convex hull for bilinear equality constraints
def convex_hull(ix, iy, iz, lx, ux, ly, uy):
    
    lhs = [cplex.SparsePair(ind = [ix, iy, iz], val = [-ly, -lx,  1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [-uy, -ux,  1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [ uy,  lx, -1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [ ly,  ux, -1]),]
    rhs = [-lx * ly, -ux * uy, lx * uy, ux * ly]
    senses = ['G', 'G', 'G', 'G']
    
    return lhs, rhs, senses


# This function defines the nonconvex MIP model
# --------------------------------------------------------------------------- #
# 
def spatial_bnb():
    
    # 1. formulate matrix
    c  = [3, 1, 5, 2, 3]
    d  = [12]
    e  = [10, 15, 5]
    A  = [[3, 1, 2, 0, 0]]
    B  = [[2, -1, 2, 1, 0], [5, 0, 2, 5, 3], [0, 2, 0, 0, 3]]
    lb = [0, 0, 0.5, 0.5, 0.5]
    ub = [1, 1, 8.0, 8.0, 8.0]
    bl = [[2, 3, 4]]
    tp = ['I', 'I', 'C', 'C', 'C']

    # 2. create cplex model
    prob = cplex.Cplex()
    prob = prob_settings(prob)

    prob.variables.add(obj = c, lb = lb, ub = ub, types = tp)
    prob.objective.set_sense(prob.objective.sense.minimize)
    # linear constraint
    M = np.concatenate((A, B), axis = 0).tolist()
    prob.linear_constraints.add(
        lin_expr = [cplex.SparsePair(ind = range(len(c)), val = r) for r in M], 
        senses = ['E'] * np.size(A, axis = 0) + ['G'] * np.size(B, axis = 0), 
        rhs = d + e)
    # convex hull
    for row in bl:
        ix, iy, iz = row[0], row[1], row[2]
        lx, ux, ly, uy = lb[ix], ub[ix], lb[iy], ub[iy]
        lhs, rhs, senses = convex_hull(ix, iy, iz, lx, ux, ly, uy)
        prob.linear_constraints.add(lin_expr = lhs, rhs = rhs, senses = senses)

    # 3. set callbacks
    contextmask = 0
    contextmask |= cplex.callbacks.Context.id.candidate
    contextmask |= cplex.callbacks.Context.id.branching
    if contextmask:
        callbacks = my_callback(lb, ub, bl)
        prob.set_callback(callbacks, contextmask)

    # 4. solve
    prob.solve()

    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", prob.solution.get_status(), ":", end = ' ')
    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print("Solution value  = ", prob.solution.get_objective_value())

    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()

    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()

    for j in range(numrows):
        print("Row %d:  Slack = %10f" % (j, slack[j]))
    for j in range(numcols):
        print("Column %d:  Value = %10f" % (j, x[j]))

    # prob.write(os.path.join(os.getcwd(), "miblp.lp"))


# Main function
# --------------------------------------------------------------------------- #
# 
if __name__ == "__main__":

    # 1. global variables
    spatial_bnb()