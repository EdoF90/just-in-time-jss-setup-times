# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import gurobipy as grb
from data_interfaces import generate_datastructures

class RepairExact():
    def __init__(self, inst, settings: dict):
        # TODO: manca il vincolo sul setup all'inizio
        # get data structure for gurobi
        due_date, release_date, final_operations, n_positions, positions, duration_op, machine_setup = generate_datastructures(inst)
        self.inst = inst
        self.machine_setup = machine_setup
        # list of constraints to update
        self.constr_to_update = []
        # create model
        self.model = grb.Model('min_tardiness')

        # Y_ij = 1 if operation i is the immediate predecessor of job j on machine m
        self.Y = self.model.addVars(
            inst.lst_operations, inst.lst_operations, inst.lst_machines,
            vtype=grb.GRB.BINARY,
            name='Y'
        )
        # V_imk = 1 if operation j is in position k for machine m
        self.V = self.model.addVars(
            inst.lst_operations, inst.lst_machines, positions,
            vtype=grb.GRB.BINARY,
            name='V'
        )
        # Earliness
        self.E = self.model.addVars(
            final_operations,
            vtype=grb.GRB.CONTINUOUS,
            name='E'
        )
        # Tardiness
        self.T = self.model.addVars(
            final_operations,
            vtype=grb.GRB.CONTINUOUS,
            name='T'
        )
        # Completion
        self.C = self.model.addVars(
            inst.lst_operations, 
            vtype=grb.GRB.CONTINUOUS,
            name='C'
        )
        # Starting
        self.S = self.model.addVars(
            inst.lst_operations,
            vtype=grb.GRB.CONTINUOUS,
            name='S'
        )

        expr = grb.quicksum(
            self.T[op] for op in final_operations
        ) #settings['tardiness_penalty'] * 
        expr += grb.quicksum(
            self.E[op] for op in final_operations
        ) #settings['earliness_penalty'] * 
        self.model.setObjective(expr, grb.GRB.MINIMIZE)

        # Earliness, tardiness definition
        self.model.addConstrs(
            self.C[op] + self.E[op] - self.T[op] == due_date[op] for op in final_operations
        )

        # Each op set in some place
        self.model.addConstrs(
            (grb.quicksum(self.V[j[0], j[1], m, k] for m in inst.eligible_machines[j] for k in positions) == 1 for j in inst.lst_operations),
            name="assigned"
        )
        # No more than one op for each slot position
        self.model.addConstrs(
            (grb.quicksum(self.V[i[0], i[1], m, k] for i in inst.machines_ops[m] ) <= 1 for k in positions for m in inst.lst_machines),
            name="limit"
        )
        # Position order
        self.model.addConstrs(
            (grb.quicksum(self.V[i[0], i[1], m, k] for i in inst.machines_ops[m]) <= grb.quicksum(self.V[i[0], i[1], m, k - 1] for i in inst.machines_ops[m]) for m in inst.lst_machines for k in range(1, n_positions)),
            name="order"
        )
        # Link Y, V
        self.model.addConstrs(
            self.V[j[0], j[1], m, k] + self.V[i[0], i[1], m, k - 1] <= 1 + self.Y[i[0], i[1], j[0], j[1], m] for i in inst.lst_operations for j in inst.lst_operations for k in range(1, n_positions) for m in inst.lst_machines
        )
        # Time inter-job
        self.model.addConstrs(
            self.C[i] >= self.S[i] + grb.quicksum(duration_op[l][i] *  self.V[i[0], i[1], l, k] for k in positions for l in inst.lst_machines) for i in inst.lst_operations
        )
        # Time intra-job
        self.model.addConstrs(
            self.S[j] >= self.C[i] + machine_setup(m, i[0], j[0]) * self.Y[i[0], i[1], j[0], j[1], m] - 1000*(1-self.Y[i[0], i[1], j[0], j[1], m]) for i in inst.lst_operations for j in inst.lst_operations for m in inst.lst_machines
        )
        # Release date
        for op in inst.lst_operations:
            self.model.addConstr(
                self.S[op] >= release_date[op[1]]
            )

        # Technological precedence
        for op in inst.lst_operations:
            lst_pred = inst.operations_forest.predecessors(op)
            for pred in lst_pred:
                self.model.addConstr(
                    self.S[op[0], op[1]] >= self.C[pred[0], pred[1]],
                    name="precedence"
                )
                data_edge = inst.operations_forest.get_edge_data(
                    pred,
                    op
                )
                if data_edge['max_waiting'] != np.inf:
                    self.model.addConstr(
                        self.S[op] <= self.S[pred] + data_edge['max_waiting']
                    )
                if data_edge['min_waiting'] != 0:
                    self.model.addConstr(
                        self.S[op] >= self.C[pred] + data_edge['min_waiting']
                    )

    def fix_assignment(self, assignment, fix_order=False, fix_assignment=False):
        for ele in self.constr_to_update:
            self.model.remove(ele)
        if fix_order:
            # METHOD 1: fixing the y:
            for machine, op_lst in assignment.sol_machines.items():
                for pos_op, op in enumerate(op_lst):
                    self.constr_to_update.append(
                        self.model.addConstr(
                            self.V[op[0], op[1], machine, pos_op] == 1,
                            name="assignmentConstr"
                        )
                    )
        if fix_assignment:
            # METHOD 2: add assignment variables:
            # X_jk = 1 if operation j is in machine m
            self.X = self.model.addVars(
                self.inst.lst_operations, self.inst.lst_machines,
                vtype=grb.GRB.BINARY,
                name='X'
            )
            # fix just the assignment variables
            for machine, op_lst in assignment.sol_machines.items():
                for op in op_lst:
                    self.constr_to_update.append(
                        self.model.addConstr(
                            self.X[op[0], op[1], machine] == 1,
                            name="assignmentConstr"
                        )
                    )
            # link the variables to the model:
            self.constr_to_update.append(
                self.model.addConstrs(
                    (self.X[op[0], op[1], m] <= grb.quicksum(self.V[op[0], op[1], m, k] for k in positions) for op in inst.lst_operations  for m in inst.lst_machines),
                    name="linkXY"
                )
            )
        self.model.update()

    def solve(self, gap=None, time_limit=None, verbose=False, lp_name=None):
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        self.model.setParam('LogFile', './logs/gurobi.log')
        if lp_name:
            self.model.write(f"./logs/{lp_name}.lp")
        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start
        if self.model.status == grb.GRB.Status.OPTIMAL:
            new_of = self.model.getObjective().getValue()
            # GETTING SOLUTIONS
            sol = []
            new_setups = []
            for op in self.inst.lst_operations:
                for m in self.inst.lst_machines:
                    if self.V.sum(op[0], op[1],m,'*').getValue() == 1:
                        machine = m
                sol.append(
                    {
                        "machine": machine,
                        "t_start": self.S[op].X,
                        "t_end": self.C[op].X,
                        "op": op
                    }
                )

            new_df_sol = pd.DataFrame.from_dict(sol)
            new_df_sol.sort_values(by=['machine', 't_start'], inplace=True)

            for machine in self.inst.lst_machines:
                tmp = new_df_sol[new_df_sol['machine'] == machine]
                initial_setting = self.inst.machines_initial_state[machine]
                if len(tmp) > 0:
                    setup_time = self.machine_setup(machine, initial_setting, tmp.iloc[0]['op'][0])
                    if setup_time > 0:
                        new_setups.append(
                            {
                                'machine': machine,
                                't_start': 0,
                                't_end': setup_time
                            }
                        )
                    for i in range(len(tmp)-1):
                        tmp_setup = self.machine_setup(machine, tmp.iloc[i]['op'][0], tmp.iloc[i+1]['op'][0])
                        if tmp_setup > 0:
                            new_setups.append(
                                {
                                    'machine': machine,
                                    't_start': self.C[tmp.iloc[i]['op'][0],tmp.iloc[i]['op'][1]].X,
                                    't_end': self.C[tmp.iloc[i]['op'][0],tmp.iloc[i]['op'][1]].X + tmp_setup
                                }
                            )

            new_assignment = -1
        else:
            print("INFEASIBLE MODEL")
            new_of = -1
            new_assignment = -1
            new_df_sol = -1
            new_setups = -1
        return new_of, new_assignment, new_df_sol, new_setups
