# -*- coding: utf-8 -*-
import time
import gurobipy as grb
from instances import *
from .exact_model import SolveJITJSSST
from solvers import *

"""
Solve neiborhood for the JITJSSST.
"""

class SolveJITJSSSTNeighborhood(SolveJITJSSST):

    def __init__(self, inst: InstanceJobShopSetUp, settings: dict = {}):
        super(SolveJITJSSSTNeighborhood, self).__init__(inst, settings)
        # initialize vector of assignment constraints:
        self.assignment_constraints = []
        
        jobs = range(inst.n_jobs)
        machines = range(inst.n_machines)

        # V_{ij}^m = 1 if operation of job i is set before op of job j in machine m
        self.V = self.model.addVars(
            self.inst.n_jobs, self.inst.n_jobs, inst.n_machines,
            vtype=grb.GRB.CONTINUOUS,
            name='V'
        )

        self.model.addConstrs(
            (self.Y[i, j, m] <= self.V[i, j, m] for i in jobs for j in jobs for m in machines),
            name="linkYV"
        )
        self.model.addConstrs(
            (self.V[i, j, m] + self.V[j, i, m] == 1 for i in jobs for j in jobs for m in machines if i > j),
            name="ijOji"
        )
        self.model.addConstrs(
            (self.V[j, j, m] == 0 for j in jobs for m in machines),
            name="noSelfPrec"
        )
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('ImproveStartTime', 0)

    def _set_initial_sol(self, assignment):
        for m, ele in assignment.items():
            for i in range(self.inst.n_jobs):
                for j in range(self.inst.n_jobs):
                    self.Y[i,j,m].start = 0
            for i in range(len(ele) - 1):
                self.Y[ele[i][1], ele[i+1][1], m].start = 1

    def allocate(self, new_assignment, pre_destroy_sol=None, verbose=False, lp_name=None, gap=None, time_limit=None):
        # No eligible machine to consider since operation can be allocated only in one machine
        # since, no flexible jss, we have just one eligible machine for each operation

        # set the initial solution get before the destroy
        if pre_destroy_sol:
            self._set_initial_sol(pre_destroy_sol)

        # REMOVE CONSTRAINTS:
        for ele in self.assignment_constraints:
            self.model.remove(ele)
        self.assignment_constraints = []

        for machine, lst_op in new_assignment.sol_machines.items():
            # If the assignment fixes all the operations in the machine:
            if len(lst_op) == len(self.inst.jobs_on_machine[machine]):
                for pos in range(0, len(lst_op) - 1):
                    self.assignment_constraints.append(
                        self.model.addConstr(
                            self.Y[lst_op[pos][1], lst_op[pos + 1][1],
                                    machine] == 1,
                            name=f"{lst_op[pos][1]}StrictlyBefore{lst_op[pos + 1][1]}Machine{machine}"
                        )
                    )
            else:
                # else just fix the loose before operator
                for pos0 in range(0, len(lst_op)):
                    # for each element, consider all the precedence op:
                    for pos1 in range(0, pos0):
                        self.assignment_constraints.append(
                            self.model.addConstr(
                                self.V[lst_op[pos1][1], lst_op[pos0][1],
                                        machine] == 1,
                                name=f"{lst_op[pos1][1]}Before{lst_op[pos0][1]}Machine_{machine}"
                            )
                        )
        start = time.time()
        self.solve(lp_name=lp_name, gap=gap, time_limit=time_limit, verbose=verbose)
        end = time.time()
        of_new = self.model.getObjective().getValue()
        
        self._get_df_sol()
        for machine in self.inst.lst_machines:
            df_tmp = self.df_sol[self.df_sol.machine == machine]            
            pos = 0
            while pos < len(df_tmp):
                ele = df_tmp.iloc[pos].op
                # if we arrive at the end of the list:
                if pos == len(new_assignment.sol_machines[machine]):
                    new_assignment.insert_pos(ele, machine, pos)
                else:
                    if new_assignment.sol_machines[machine][pos] != ele:
                        # if the operation is removed:
                        if ele in new_assignment.removed_ops:
                            new_assignment.insert_pos(ele, machine, pos)
                        else:
                            # else the op was moved:
                            new_assignment.move_op_within(ele, pos=pos)
                            pos += 1
                    else:
                        pos += 1 
        if verbose: print(f">>> NEW OF: {of_new} - {end - start}")
        info = {
            "comp_time": end - start,
            "mip_gap": self.model.MIPGap,
            "tardiness": self.tardiness.getValue(),
            "earliness": self.earliness.getValue(),
            "flow_time": self.flow_time.getValue(),
        }
        return of_new, self.df_sol, info
