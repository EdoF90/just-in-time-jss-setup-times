# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import gurobipy as grb
from instances import InstanceJobShop
from solvers.solution_classes import SchedulingAssignment

"""
Solve the exact model, it is the same as SolveJITJSSST, if you do not fix any variable.
"""

def solve_jit_jss_setup_times(inst:InstanceJobShop, gap=None, time_limit=None, verbose=False):
    
    jobs = range(inst.n_jobs)
    machines = range(inst.n_machines)
    n_ops = inst.n_machines
    operations = range(n_ops) 

    model = grb.Model('jit_jss_setup_times')
    # Earliness
    E = model.addVars(
        inst.n_jobs,
        vtype=grb.GRB.CONTINUOUS,
        name='E'
    )
    # Tardiness
    T = model.addVars(
        inst.n_jobs,
        vtype=grb.GRB.CONTINUOUS,
        name='T'
    )
    # Completion
    C = model.addVars(
        n_ops, inst.n_jobs,
        vtype=grb.GRB.CONTINUOUS,
        name='C'
    )
    # Starting Time
    S = model.addVars(
        n_ops, inst.n_jobs,
        vtype=grb.GRB.CONTINUOUS,
        name='S'
    )
    # Precedence on machine
    Y = model.addVars(
        inst.n_jobs,inst.n_jobs,inst.n_machines,
        vtype=grb.GRB.BINARY,
        name='Y'
    )

    # OBJECTIVE FUNCTION
    expr = grb.quicksum(
        inst.df_jobs.iloc[j].earliness_penalty * T[j] for j in jobs
    )
    expr += grb.quicksum(
        inst.df_jobs.iloc[j].tardiness_penalty * E[j] for j in jobs
    )
    expr += grb.quicksum(
        inst.df_jobs.iloc[j].tardiness_penalty * E[j] for j in jobs
    )
    model.setObjective(expr, grb.GRB.MINIMIZE)
    
    model.addConstrs(
        (C[n_ops - 1, j] - T[j] + E[j] == inst.df_jobs.iloc[j].due_date for j in jobs),
        name="earlinessTardinessDef"
    )
    
    model.addConstrs(
        (S[i, j] >= C[i - 1, j] for j in jobs for i in range(1, n_ops)),
        name="sequenceOpsSameJob"
    )
    model.addConstrs(
        (S[inst.op_from_j_m[j1,m], j1] >= C[inst.op_from_j_m[j0,m], j0] + inst.get_setup(m, j0, j1) - 10 * max(inst.df_jobs.due_date) * (1 - Y[j0, j1, m])  for m in machines for j0 in jobs for j1 in jobs if j0!=j1),
        name="setupConstr"
    )

    model.addConstrs(
        (C[i, j] == S[i, j] + inst.lst_job[j][i]['processing_time'] for j in jobs for i in operations),
        name="opDuration"
    )

    model.addConstrs(
        (S[0, j] >= inst.df_jobs.iloc[j].release_date for j in jobs),
        name="releaseDate"
    )

    model.addConstrs(
        (Y[i, j, m] + Y[j, i, m] == 1
        for i in jobs
        for j in jobs
        for m in machines
        if i > j),
        name="i_before_j_or_reverse"
    )

    model.update()
    if gap:
        model.setParam('MIPgap', gap)
    if time_limit:
        model.setParam(grb.GRB.Param.TimeLimit, time_limit)
    if verbose:
       model.setParam('OutputFlag', 1)
    else:
        model.setParam('OutputFlag', 0)
    model.setParam('LogFile', './logs/gurobi.log')
    model.write("./logs/jit_jss_setup.lp")
    start = time.time()
    model.optimize()
    end = time.time()
    comp_time = end - start
    if model.status == grb.GRB.Status.OPTIMAL:
        sol = []
        setups = []
        of = model.getObjective().getValue()

        for j, job in enumerate(inst.lst_job):
            for pos, detail in enumerate(job):
                # print(f"S[{pos},{j}] {S[pos,j].X} C[{pos},{j}]: {C[pos,j].X}")
                sol.append(
                    {
                        "machine": detail['machine'],
                        "t_start": S[pos,j].X,
                        "t_end": C[pos,j].X,
                        "op": (pos,j)
                    }
                )

        df_sol = pd.DataFrame.from_dict(sol)
        df_sol.sort_values(by=['machine', 't_start'], inplace=True)
        print(df_sol)
                        
        setups = []
        for m in range(inst.n_machines):
            for i in range(inst.n_jobs):
                for j in range(inst.n_jobs):
                    if Y[i,j,m].X > 0.5:
                        setup_time = inst.get_setup(m, i, j)
                        if setup_time > 0:
                            setups.append(
                                {
                                    'machine': m,
                                    't_start': C[inst.op_from_j_m[i,m],i].X,
                                    't_end': C[inst.op_from_j_m[i,m],i].X + setup_time,
                                    'from': i,
                                    'to': j
                                }
                            )
        return of, df_sol, setups, comp_time
    else:
        return np.Inf, [], [], comp_time


"""
Solve the exact model, it is the same as SolveJITJSSST, if you fix all the variables.
"""
class SolveTimingJITJSSST():
    def __init__(self, inst:InstanceJobShop, settings:dict={}):
        self.inst = inst
        self.settings = settings
        
        # Timing model of jit_jss_setup_times
        self.jobs = range(inst.n_jobs)
        self.machines = range(inst.n_machines)
        n_ops = inst.n_machines
        operations = range(n_ops) 
        self.model = grb.Model('timing_model_jit_jss_setup_times')
        # Earliness
        self.E = self.model.addVars(
            inst.n_jobs,
            vtype=grb.GRB.CONTINUOUS,
            name='E'
        )
        # Tardiness
        self.T = self.model.addVars(
            inst.n_jobs,
            vtype=grb.GRB.CONTINUOUS,
            name='T'
        )
        tuple_for_ops = [ (i, j) for j in self.jobs for i in range(inst.n_ops[j])]
        # Completion
        self.C = self.model.addVars(
            tuple_for_ops,
            vtype=grb.GRB.CONTINUOUS,
            name='C'
        )
        # Starting Time
        self.S = self.model.addVars(
            tuple_for_ops,
            vtype=grb.GRB.CONTINUOUS,
            name='S'
        )

        # OBJECTIVE FUNCTION
        self.tardiness = grb.quicksum(
            inst.jobs[j]['tardiness_penalty'] * self.T[j] for j in self.jobs
        )
        self.earliness = grb.quicksum(
            inst.jobs[j]['earliness_penalty'] * self.E[j] for j in self.jobs
        )
        self.flow_time = grb.quicksum(
            inst.jobs[j]['flow_time_penalty'] * (self.C[inst.n_ops[j] - 1, j] - self.S[0, j]) for j in self.jobs
        )
        self.model.setObjective(self.tardiness + self.earliness + self.flow_time, grb.GRB.MINIMIZE)
        
        self.model.addConstrs(
            (self.C[inst.n_ops[j] - 1, j] - self.T[j] + self.E[j] == inst.df_jobs.iloc[j].due_date for j in self.jobs),
            name="earlinessTardinessDef"
        )
        
        self.model.addConstrs(
            (self.S[i, j] >= self.C[i - 1, j] for j in self.jobs for i in range(1, inst.n_ops[j])),
            name="sequenceOpsSameJob"
        )

        self.model.addConstrs(
            (self.C[i, j] == self.S[i, j] + inst.lst_job[j][i]['processing_time'] for j in self.jobs for i in range(inst.n_ops[j])),
            name="opDuration"
        )

        self.model.addConstrs(
            (self.S[0, j] >= inst.df_jobs.iloc[j].release_date for j in self.jobs),
            name="releaseDate"
        )
        self.first_run = True
        self.precedence_constraints = []

    def _run_optimization(self, assignment: SchedulingAssignment, gap=None, time_limit=None, verbose=False, file_path=None):
        sol_machines = assignment.sol_machines
        if not self.first_run:
            for ele in self.precedence_constraints:
                self.model.remove(ele)
            self.model.update()
            self.precedence_constraints = []
            
        self.first_run = False
        # Assignment Constraints
        for m in self.machines:
            self.precedence_constraints.append(
                self.model.addConstrs(
                    (
                        self.S[sol_machines[m][i+1]] >= self.C[sol_machines[m][i]] + self.inst.get_setup(
                            m,
                            sol_machines[m][i][1],
                            sol_machines[m][i+1][1]
                        )
                        for i in range(len(sol_machines[m]) - 1)
                    ),
                    name="setupConstr"
                )
            )
        self.model.update()

        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        self.model.setParam('LogFile', './logs/gurobi.log')
        if file_path:
            self.model.write(file_path)
        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start
        if self.model.status == grb.GRB.Status.OPTIMAL:
            of = self.model.getObjective().getValue()
        else:
            of = np.inf
        return of, comp_time

    def _get_solution(self, assignment: SchedulingAssignment):
        sol = []
        setups = []
        for machine, operations_tmp in assignment.sol_machines.items():
            for i, op in enumerate(operations_tmp):
                earliness = 0
                tardiness = 0
                idx_job = op[1]
                if op[0] == self.inst.n_machines - 1:
                    earliness = self.E[op[0]].X
                    tardiness = self.T[op[0]].X
                sol.append(
                    {
                        "machine": machine,
                        "t_start": self.S[op].X,
                        "t_end": self.C[op].X,
                        "op": op,
                        "order": idx_job,
                        "earliness": earliness,
                        "tardiness": tardiness
                    }
                )
                if i < len(operations_tmp) - 1:
                    setup_time = self.inst.get_setup(
                        machine,
                        idx_job,
                        operations_tmp[i+1][1]
                    )
                    if setup_time > 0:
                        setups.append(
                            {
                                "machine": machine,
                                "t_start": self.C[op].X,
                                "t_end": self.C[op].X + setup_time,
                            }
                        )

        df_sol = pd.DataFrame.from_dict(sol)
        return df_sol, setups
