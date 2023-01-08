# --------------------------------------------------------------------------- #
# MIT license
# Copyright 2020–2021, School of Electrical Engineering, Southeast University, 
# Pengxiang Liu. All Rights Reserved.
# 
# File: spatial-bnb-cplex-1.py
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

In this script, the sB&B algorithm is implemented in CPLEX through generic 
callback interface. Spatial branching is implemented on the variable of the 
first bilinear infeasible constraint using classical branching strategy.

min   c * x
s.t.  A * x == d
      B * x >= e
      lb <= x <= ub
      x(p_n) * x(q_n) = x(r_n)
'''


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

    # turn off presolve
    prob.parameters.threads.set(1)
    prob.parameters.preprocessing.reduce.set(0)
    prob.parameters.preprocessing.linear.set(0)
    
    # return
    return prob


# add the convex hull for bilinear equality constraints
# --------------------------------------------------------------------------- #
# 
def convex_hull(ix, iy, iz, lx, ux, ly, uy):
    
    lhs = [cplex.SparsePair(ind = [ix, iy, iz], val = [-ly, -lx,  1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [-uy, -ux,  1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [ uy,  lx, -1]),
           cplex.SparsePair(ind = [ix, iy, iz], val = [ ly,  ux, -1]),]
    rhs = [-lx * ly, -ux * uy, lx * uy, ux * ly]
    senses = ['G', 'G', 'G', 'G']
    
    return lhs, rhs, senses


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

    # 3. set callbacks
    contextmask = 0
    contextmask |= cplex.callbacks.Context.id.candidate
    contextmask |= cplex.callbacks.Context.id.branching
    if contextmask:
        callbacks = my_callback(lb, ub, bl)
        prob.set_callback(callbacks, contextmask)

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