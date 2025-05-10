# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import gurobipy as grb
from solvers.abs_solver import Solver
from instances import InstanceJobShopSetUp
from solvers.solution_classes import SchedulingAssignment


class SolveJITJSSST(Solver):
    def __init__(self, inst: InstanceJobShopSetUp, settings: dict = {}):
        super(SolveJITJSSST, self).__init__(inst, settings)
        self.inst = inst

        self.assignment_constraints = []
        jobs = range(inst.n_jobs)
        machines = range(inst.n_machines)

        self.model = grb.Model('jit_jss_setup_times')
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
        tuple_for_ops = [ (i, j) for j in jobs for i in range(inst.n_ops[j])]
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
        # 1 if immediate predecessors on machine
        self.Y = self.model.addVars(
            inst.n_jobs, inst.n_jobs, inst.n_machines,
            vtype=grb.GRB.BINARY,
            name='Y'
        )

        # OBJECTIVE FUNCTION
        self.tardiness = grb.quicksum(
            inst.jobs[j]['tardiness_penalty'] * self.T[j] for j in jobs
        )
        self.earliness = grb.quicksum(
            inst.jobs[j]['earliness_penalty'] * self.E[j] for j in jobs
        )
        self.flow_time = grb.quicksum(
            inst.jobs[j]['flow_time_penalty'] * (self.C[inst.n_ops[j] - 1, j] - self.S[0, j])
            for j in jobs
        )
        self.model.setObjective(self.tardiness + self.earliness + self.flow_time, grb.GRB.MINIMIZE)

        for j in jobs:
            self.model.addConstr(
                self.C[inst.n_ops[j] - 1, j] - self.T[j] + self.E[j] ==
                inst.df_jobs.iloc[j].due_date,
                name=f"earlinessTardinessDef_{j}"
            )
        
        for m in machines:
            self.model.addConstrs(
                (self.S[inst.op_from_j_m[j1, m], j1] >= self.C[inst.op_from_j_m[j0, m], j0] + inst.get_setup(m, j0, j1) - 10 *
                max(inst.df_jobs.due_date) * (1 - self.Y[j0, j1, m]) for j0 in inst.jobs_on_machine[m] for j1 in inst.jobs_on_machine[m] if j0 != j1),
                name="setupConstr"
            )

        for j in jobs:
            self.model.addConstrs(
                (self.S[i, j] >= self.C[i - 1, j]
                for i in range(1, inst.n_ops[j])),
                name=f"sequenceOpsSameJob_{j}"
            )

        self.model.addConstrs(
            (self.S[0, j] >= inst.jobs[j]['release_date'] for j in jobs),
            name="releaseDateCase1"
        )

        self.model.addConstrs(
            (
                self.S[0, j] >= grb.quicksum(
                    inst.get_setup(m, inst.machines_initial_state[m], j) * (
                        1 - grb.quicksum(self.Y[i, j, m] for i in jobs if i != j))
                    for m in machines
                )
                for j in jobs),
            name="releaseDateCase2"
        )

        for j in jobs:
            self.model.addConstrs(
                (self.C[i, j] == self.S[i, j] + inst.lst_job[j][i]
                ['processing_time'] for i in range(inst.n_ops[j])),
                name=f"opDuration_{j}"
            )

        for m in machines:
            if m == 0:
                self.AA = self.model.addConstr(
                    grb.quicksum(self.Y[i, j, m] for i in inst.jobs_on_machine[m] for j in inst.jobs_on_machine[m] if i != j)
                    == len(inst.jobs_on_machine[m]) - 1,
                    name=f"assign_all_jobs_{m}"
                )
            else:
                self.model.addConstr(
                    grb.quicksum(self.Y[i, j, m] for i in inst.jobs_on_machine[m] for j in inst.jobs_on_machine[m] if i != j)
                    == len(inst.jobs_on_machine[m]) - 1,
                    name=f"assign_all_jobs_{m}"
                )
            self.model.addConstrs(
                (grb.quicksum(self.Y[i, j, m] for j in inst.jobs_on_machine[m])
                <= 1 for i in inst.jobs_on_machine[m]),
                name=f"max_one_after_{m}"
            )
            self.model.addConstrs(
                (grb.quicksum(self.Y[j, i, m] for j in inst.jobs_on_machine[m])
                <= 1 for i in inst.jobs_on_machine[m]),
                name=f"max_one_before_{m}"
            )
        self.force_change_constraint = None

    def perturbe_of(self, machine_assignment=None, df_sol=None, seed=None):
        if seed:
            np.random.seed(seed)
            rnd_coeff = seed * np.random.uniform(
                low=0.5,high=1.5,size=self.inst.n_jobs,
            )
        else:
            rnd_coeff = np.random.uniform(
                low=0.5,high=1.5,size=self.inst.n_jobs,
            )
        if df_sol is not None:
            # Punish late or early jobs
            tardy_order = df_sol.loc[df_sol['tardiness'].idxmax()].order
            early_order = df_sol.loc[df_sol['earliness'].idxmax()].order
            expr = self.T[tardy_order] + self.E[early_order]
        else:
            expr = grb.quicksum(
                rnd_coeff[j] * self.inst.df_jobs.iloc[j].tardiness_penalty * self.T[j] for j in range(self.inst.n_jobs)
            )
            expr += grb.quicksum(
                rnd_coeff[j] * self.inst.df_jobs.iloc[j].earliness_penalty * self.E[j] for j in range(self.inst.n_jobs)
            )
        self.model.setObjective(expr, grb.GRB.MINIMIZE)
        if machine_assignment is not None:
            # Force a different solution requiring a difference in the solution
            for m, ele in machine_assignment.items():
                expr = 0
                for i in range(self.inst.n_jobs):
                    for j in range(self.inst.n_jobs):
                        self.Y[i,j,m].start = 0
                        expr += self.Y[i,j,m]
                for i in range(len(ele) - 1):
                    self.Y[ele[i][1], ele[i+1][1], m].start = 1
                    expr += (1 - self.Y[ele[i][1], ele[i+1][1], m])
            self.force_change_constraint = self.model.addConstr(
                expr >= 10
            )

    def restore_of(self):
        expr = grb.quicksum(
            self.inst.jobs[j]['tardiness_penalty'] * self.T[j] for j in range(self.inst.n_jobs)
        )
        expr += grb.quicksum(
            self.inst.jobs[j]['earliness_penalty'] * self.E[j] for j in range(self.inst.n_jobs)
        )
        for j in range(self.inst.n_jobs):
            expr += self.inst.jobs[j]['flow_time_penalty'] * (self.C[self.inst.n_ops[j] - 1, j] - self.S[0, j])
        self.model.setObjective(expr, grb.GRB.MINIMIZE)
        if self.force_change_constraint:
            self.model.remove(self.force_change_constraint)
            self.force_change_constraint = None

    def _remove_assignment(self): 
        # TODO: it is not assignment but sequencing
        for ele in self.assignment_constraints:
            self.model.remove(ele)

    def set_assignment(self, assignment: SchedulingAssignment):
        self._remove_assignment()

        for machine, lst_op in assignment.sol_machines.items():
            for pos in range(0, len(lst_op) - 1):
                self.assignment_constraints.append(
                    self.model.addConstr(
                        self.Y[lst_op[pos][1], lst_op[pos + 1]
                               [1], machine] == 1,
                        name=f"j{lst_op[pos][1]}_before_j{lst_op[pos + 1][1]}_on_machine_{machine}"
                    )
                )

    def set_destroyed_assignment(self, assignment: SchedulingAssignment, pre_destroy_sol:dict):
        self._remove_assignment()
        # Computing precendence operations and setting initial solution
        constraints = {}
        for m, ele in pre_destroy_sol.items():
            constraints[m] = []
            for i in range(self.inst.n_jobs):
                for j in range(self.inst.n_jobs):
                    self.Y[i,j,m].start = 0
            for i in range(len(ele) - 1):
                # if both are not removed
                if (ele[i+1] not in assignment.removed_ops) and (ele[i] not in assignment.removed_ops): 
                    constraints[m].append(
                        (ele[i][1], ele[i+1][1])
                    )
                self.Y[ele[i][1], ele[i+1][1], m].start = 1

        for machine, lst_couple_op in constraints.items():
            for couple_op in lst_couple_op:
                self.assignment_constraints.append(
                    self.model.addConstr(
                        self.Y[couple_op[0], couple_op[1], machine] == 1,
                        name=f"d_j{couple_op[0]}_before_j{couple_op[1]}_on_machine_{machine}"
                    )
                )

    def solve(self, gap=None, time_limit=None, verbose=False, lp_name=None, MIPFocus=None, ImproveStartTime=None):
        self.model.update()
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if lp_name:
            self.model.write(f"./logs/{lp_name}.lp")

        if MIPFocus: 
            self.model.setParam('MIPFocus', MIPFocus)
        if ImproveStartTime:
            self.model.setParam('ImproveStartTime', ImproveStartTime)

        self.model.setParam('LogFile', './logs/gurobi.log')
        start = time.time()
        self.model.optimize()
        end = time.time()
        self.comp_time = end - start

    def _get_df_sol(self):
        if self.model.SolCount >= 1:
            sol = []
            for n_job, job in enumerate(self.inst.lst_job):
                for n_op, detail in enumerate(job):
                    earliness = 0
                    tardiness = 0
                    # ADD EARLINESS ONLY TO LAST OP
                    if n_op == len(job) - 1:
                        earliness = self.E[n_job].X
                        tardiness = self.T[n_job].X
                    sol.append(
                        {
                            "machine": detail['machine'],
                            "t_start": self.S[n_op, n_job].X,
                            "t_end": self.C[n_op, n_job].X,
                            "op": (n_op, n_job),
                            "order": n_job,
                            "earliness": earliness,
                            "tardiness": tardiness
                        }
                    )
            self.df_sol = pd.DataFrame.from_dict(sol)
            # sort_values, so that for the timing problem is easier get the data
            self.df_sol.sort_values(by=['machine', 't_start'], inplace=True)

    def get_assignment(self):
        if self.model.SolCount >= 1:
            self._get_df_sol()
            machine_assignment = {key: [] for key in self.inst.lst_machines}
            for machine in self.inst.lst_machines:
                df_tmp = self.df_sol[self.df_sol.machine == machine]
                for i in range(len(df_tmp)):
                    machine_assignment[machine].append(
                        df_tmp.iloc[i].op
                    )
            return self.model.getObjective().getValue(), SchedulingAssignment(machine_assignment), self.df_sol
        else:
            return np.Inf, None, None

    def _set_initial_sol(self, machine_assignment: dict):
        for m, ele in machine_assignment.items():
            for i in range(self.inst.n_jobs):
                for j in range(self.inst.n_jobs):
                    self.Y[i,j,m].start = 0
            for i in range(len(ele) - 1):
                self.Y[ele[i][1], ele[i+1][1], m].start = 1

    def get_solution(self):
        if self.model.SolCount >= 1:
            setups = []
            self._get_df_sol()
            setups = []
            for m in range(self.inst.n_machines):
                for i in range(self.inst.n_jobs):
                    for j in range(self.inst.n_jobs):
                        if self.Y[i, j, m].X > 0.5:
                            setup_time = self.inst.get_setup(m, i, j)
                            if setup_time > 0:
                                setups.append(
                                    {
                                        'machine': m,
                                        't_start': self.C[self.inst.op_from_j_m[i, m], i].X,
                                        't_end': self.C[self.inst.op_from_j_m[i, m], i].X + setup_time,
                                        'from': i,
                                        'to': j
                                    }
                                )
            info= {
                "comp_time": self.comp_time,
                "mip_gap": self.model.MIPGap
            }
            return self.model.getObjective().getValue(), self.df_sol, setups, info
        else:
            info= {
                "mip_gap": -1,
                "comp_time": self.comp_time
            }
            return np.Inf, [], [], info

    def compute_timing(self, assignment: SchedulingAssignment, lp_name=None, gap=None, time_limit=None, verbose=False):
        self.set_assignment(assignment)
        self.solve(verbose=verbose, lp_name=lp_name, gap=gap, time_limit=time_limit)
        return self.get_solution()

