# -*- coding: utf-8 -*-
import numpy as np
from solvers.solution_classes import SchedulingAssignment


def generate_rnd_solution(inst):
    # random generate a solution 
    # add j a number of times equal to the operation of job j
    sol = []
    for j in range(inst.n_jobs):
        sol.extend([j] * inst.n_ops[j])
    # shuffle the list, this is the order to do the jobs
    np.random.shuffle(sol)
    # create dict sol_machines
    sol_machines = {key: [] for key in inst.lst_machines}
    # associate operations to machine as in list scheduling. 
    idx_op = [0] * inst.n_jobs
    for idx_job in sol:
        op = (idx_op[idx_job], idx_job)
        data_op = inst.df_operations[inst.df_operations.op == op].iloc[0]
        sol_machines[data_op['machines']].append(op)
        idx_op[idx_job] += 1
    return SchedulingAssignment(sol_machines)
