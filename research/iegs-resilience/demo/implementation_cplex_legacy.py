# --------------------------------------------------------------------------- #
# MIT license
# Copyright 2020–2021, School of Electrical Engineering, Southeast University, 
# Pengxiang Liu. All Rights Reserved.
# 
# File: implementation_cplex_legacy.py
# Version: 1.0.0
# --------------------------------------------------------------------------- #


'''
A spatial branch-and-bound (sB&B) demo script for non-convex problem where the 
model is formulated as a mixed-integer bilinear programming.

min   3 * x0 + 1 * x1 + 5 * y0 + 2 * y1 + 3 * y2
s.t.  3 * x0 + 1 * x1 + 2 * y0 = 12
      2 * x0 - 1 * x1 + 2 * y0 + 2 * y1 >= 10
      5 * x1 + 2 * y0 + 5 * y1 + 3 * y2 >= 15
      2 * x1 + 3 * y2 >= 5
      y0 * y1 = y2 (bilinear constraint)
      0.5 <= y0 <= 8, 0.5 <= y1 <= 8, 0.5 <= y2 <= 8
      x0 and x1 are binary; y0, y1 and y2 are continuous

In this script, the sB&B algorithm is implemented in CPLEX through legacy 
callback interface. Spatial branching is implemented on the variable of the 
first bilinear infeasible constraint using classical branching strategy.

min   c * x
s.t.  A * x == d
      B * x >= e
      lb <= x <= ub
      x(p_n) * x(q_n) = x(r_n)
'''


import os, sys, cplex
import numpy as np

from scipy.sparse import coo_matrix


# incumbent callback
# --------------------------------------------------------------------------- #
# 
class incumbent_callback(cplex.callbacks.IncumbentCallback):
    
    # initialize
    def __init__(self, env):
        super().__init__(env)
        self.num_called = 0
    
    # define the call function
    def __call__(self):
        # 1) get information
        self.num_called = self.num_called + 1
        var, obj = self.get_values(), self.get_objective_value()
        # 2) checking
        if not is_bilinear_feasible(self.coef, var):
            self.reject()


# branching callback
# --------------------------------------------------------------------------- #
# 
class branching_callback(cplex.callbacks.BranchCallback):
    
    # initialize
    def __init__(self, env):
        super().__init__(env)
        self.num_called = 0
    
    # define the call function
    def __call__(self):
        # 1) get information
        self.num_called = self.num_called + 1
        var, obj = self.get_values(), self.get_objective_value()
        # 2) checking
        if is_bilinear_feasible(self.coef, var):
            return
        # 3) spatial branching
        lb = self.get_lower_bounds()
        ub = self.get_upper_bounds()
        for i, row in enumerate(self.coef['bl']):
            err = var[row[0]] * var[row[1]] - var[row[2]]
            index, value = row[0], var[row[0]]
            if err >= 1e-6:
                self.spatial_branching(row, index, value, lb, ub, 'L')
                self.spatial_branching(row, index, value, lb, ub, 'U')
    
    # create branches
    def spatial_branching(self, row, index, value, lb, ub, child):
        ip, iq, ir = row[0], row[1], row[2]
        lp, up, lq, uq = lb[ip], ub[ip], lb[iq], ub[iq]
        if ip == index and child == 'L': lp = value
        if ip != index and child == 'L': lq = value
        if ip == index and child == 'U': up = value
        if ip != index and child == 'U': uq = value
        lhs, rhs, sense = self.convex_hull(ip, iq, ir, lp, up, lq, uq)
        self.make_branch(
            self.get_objective_value(),
            variables = [(index, child, value)], 
            constraints = [(lhs[i], sense[i], rhs[i]) for i in range(len(lhs))]
        )

    # create constraints for convex hull
    def convex_hull(self, ix, iy, iz, lx, ux, ly, uy):
        # pre-processing
        ix, iy, iz = int(ix), int(iy), int(iz)
        # build constraints
        if ix == iy:
            lhs = [cplex.SparsePair(ind = [ix, iz], val = [-2 * lx,  1]),
                   cplex.SparsePair(ind = [ix, iz], val = [-2 * ux,  1]),
                   cplex.SparsePair(ind = [ix, iz], val = [ux + lx, -1])]
            rhs = [-lx * lx, -ux * ux, lx * ux]
            senses = ['G', 'G', 'G']
        else:
            lhs = [cplex.SparsePair(ind = [ix, iy, iz], val = [-ly, -lx,  1]),
                   cplex.SparsePair(ind = [ix, iy, iz], val = [-uy, -ux,  1]),
                   cplex.SparsePair(ind = [ix, iy, iz], val = [ uy,  lx, -1]),
                   cplex.SparsePair(ind = [ix, iy, iz], val = [ ly,  ux, -1])]
            rhs = [-lx * ly, -ux * uy, lx * uy, ux * ly]
            senses = ['G', 'G', 'G', 'G']
        # return
        return lhs, rhs, senses


# check bilinear feasibility
# --------------------------------------------------------------------------- #
# 
def is_bilinear_feasible(coef, var):
    
    err = np.zeros((len(coef['bl']), 1))
    for i, row in enumerate(coef['bl']):
        p_n, q_n, r_n = row[0], row[1], row[2]
        err[i] = var[p_n] * var[q_n] - var[r_n]
    if np.max(err) >= 1e-6:
        return False
    else:
        return True


# set coefficient matrices
# --------------------------------------------------------------------------- #
# 
def set_coefficient():

    c  = [3, 1, 5, 2, 3]
    d  = [12]
    e  = [10, 15, 5]
    A  = [[3,  1,  2,  0,  0]]
    B  = [[2, -1,  2,  1,  0], 
          [5,  0,  2,  5,  3], 
          [0,  2,  0,  0,  3]]
    lb = [0, 0, 0.5, 0.5, 0.5]
    ub = [1, 1, 8.0, 8.0, 8.0]
    bl = [[2, 3, 4]]
    tp = ['I', 'I', 'C', 'C', 'C']

    name = 'c, d, e, A, B, lb, ub, bl, tp'.split((', '))
    coef = {}
    for i in name:
        coef[i] = eval(i)
    
    return coef


# cplex parameters
# --------------------------------------------------------------------------- #
# 
def set_cplex_parameters(prob):

    # 1. settings
    # 1.1) mip
    prob.parameters.mip.tolerances.mipgap.set(1e-4)
    prob.parameters.mip.tolerances.integrality.set(1e-8)
    prob.parameters.mip.strategy.heuristicfreq.set(-1)
    # 1.2) pre-processing
    prob.parameters.preprocessing.reduce.set(1)
    prob.parameters.preprocessing.aggregator.set(0)
    prob.parameters.preprocessing.reformulations.set(0)
    # 1.3) time limit
    prob.parameters.timelimit.set(600)

    # 2. return
    return prob


# build model
# --------------------------------------------------------------------------- #
# 
def set_cplex_model(coef):

    # 1. initialize cplex class
    # 1.1) cplex 
    prob = cplex.Cplex()
    prob = set_cplex_parameters(prob)
    # 1.2) coefficient
    c  = coef['c' ]
    d  = coef['d' ]
    e  = coef['e' ]
    A  = coef['A' ]
    B  = coef['B' ]
    lb = coef['lb']
    ub = coef['ub']
    bl = coef['bl']
    tp = coef['tp']

    # 2. add variables and constraints
    # 2.1) variables
    prob.variables.add(obj = c, lb = lb, ub = ub, types = tp)
    prob.objective.set_sense(prob.objective.sense.minimize)
    # 2.2) linear constraint
    for i in range(len(A)):
        A_ind, A_val = coo_matrix(A[i]).col, coo_matrix(A[i]).data
        expr = cplex.SparsePair(ind = A_ind.tolist(), val = A_val.tolist())
        prob.linear_constraints.add(
            lin_expr = [expr], senses = ['E'], rhs = [d[i]]
        )
    for i in range(len(B)):
        B_ind, B_val = coo_matrix(B[i]).col, coo_matrix(B[i]).data
        expr = cplex.SparsePair(ind = B_ind.tolist(), val = B_val.tolist())
        prob.linear_constraints.add(
            lin_expr = [expr], senses = ['G'], rhs = [e[i]]
        )
    
    # 3. callbacks
    # 3.1) incumbent callback
    incumbent = prob.register_callback(incumbent_callback)
    incumbent.coef = coef
    # 3.2) branch callback
    branching = prob.register_callback(branching_callback)
    branching.coef = coef

    # 4. return
    return prob


# script entrance
# --------------------------------------------------------------------------- #
# 
if __name__ == "__main__":

    # 1. create models
    coef = set_coefficient()
    prob = set_cplex_model(coef)

    # 2. solve
    prob.solve()

    # 3. print results
    print()
    # 3.1) return an integer code
    print("Solution status = ", prob.solution.get_status(), ":", end = ' ')
    # 3.1) print the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print("Solution value  = ", prob.solution.get_objective_value())
    # 3.3) constraints and variables
    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()
    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()
    for j in range(numrows):
        print("Row %d:  Slack = %10f" % (j, slack[j]))
    for j in range(numcols):
        print("Column %d:  Value = %10f" % (j, x[j]))