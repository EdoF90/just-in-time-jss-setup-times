# -*- coding: utf-8 -*-
import math
import logging
import numpy as np
import gurobipy as grb
from instances import Instance
from solvers.just_in_time_job_shop_setup.exact_model import SolveJITJSSST
from solvers.just_in_time_job_shop_setup.solve_timing import SolveTimingJITJSSST
from solvers.solution_classes import SchedulingAssignment


class DestroyToolkit():
    def __init__(self, inst: Instance, OP_TO_DESTROY: int = {}):
        self.inst = inst
        self.OP_TO_DESTROY = OP_TO_DESTROY
        self.solver = SolveJITJSSST(inst)
        self.solver_timing = SolveTimingJITJSSST(inst)
    
    def set_n_op_to_destroy(self, new_op_to_destroy):
        self.OP_TO_DESTROY = new_op_to_destroy

    def random_jobs(self, assignment:SchedulingAssignment, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_jobs_to_destroy = math.ceil(self.OP_TO_DESTROY / self.inst.n_machines)
        id_job_to_destroy = np.random.choice(
            df_sol.order.unique(), n_jobs_to_destroy, replace=False,
        )
        logging.info(f"\t random_jobs: {id_job_to_destroy}")
        self._remove_jobs(assignment, id_job_to_destroy)

    def _remove_jobs(self, assignment:SchedulingAssignment, id_job_to_destroy):
        for _, ele in self.inst.df_operations[self.inst.df_operations.order.isin(id_job_to_destroy)].iterrows():
            assignment.remove_op(ele.op)

    def random_ops(self, assignment: SchedulingAssignment, seed=None):
        if seed is not None:
            np.random.seed(seed)
        logging.info("\t random_ops")
        # print("\t random_ops")
        # count the number of ops for each machine
        n_ops = {key: len(assignment.sol_machines[key]) for key in self.inst.lst_machines}
        for _ in range(self.OP_TO_DESTROY):
            # get a vector of probabilities from n_ops
            tmp_prob = np.array([ele for _, ele in n_ops.items()])
            tmp_prob = tmp_prob / sum(tmp_prob)
            # sample a machine
            rnd_machine = np.random.choice(
                self.inst.lst_machines,
                p=tmp_prob
            )
            # sample one operation
            rnd_pos = np.random.randint(
                0,
                n_ops[rnd_machine]
            )
            # remove the entry from the dict
            assignment.remove_machine_pos(rnd_machine, rnd_pos)
            n_ops[rnd_machine] -= 1

    def worst_jobs(self, assignment:SchedulingAssignment, seed=None):
        _, df_sol, _, _ = self.solver.compute_timing(
            assignment
        )
        if seed is not None:
            np.random.seed(seed)
        logging.info("\t worst_jobs")
        # print("\t worst_jobs")
        id_most_tardy_job = df_sol.loc[df_sol['tardiness'].idxmax()].order
        id_most_early_job = df_sol.loc[df_sol['earliness'].idxmax()].order
        if id_most_early_job == id_most_tardy_job:
            # if they are equal change randomly one work.
            id_most_early_job = np.random.choice(
                df_sol[df_sol.order != id_most_tardy_job].order,
                1
            )[0]
        for _, ele in df_sol[df_sol.order == id_most_tardy_job].iterrows():
            assignment.remove_op(ele.op)
        for _, ele in df_sol[df_sol.order == id_most_early_job].iterrows():
            assignment.remove_op(ele.op)

    def random_machines(self, assignment:SchedulingAssignment, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # print("\t random_machines")
        n_machine_to_destroy = math.ceil(self.OP_TO_DESTROY / self.inst.n_jobs)
        rnd_machines = np.random.choice(
            self.inst.lst_machines,
            n_machine_to_destroy,
            replace=False
        )
        logging.info(f"\t random_machines: {rnd_machines}")
        for m in rnd_machines:
            assignment.remove_machine(m)

    def random_slice(self, assignment:SchedulingAssignment, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_op_to_remove_per_machine = math.ceil(self.OP_TO_DESTROY / self.inst.n_machines)
        max_start_slice = len(assignment.sol_machines[0]) - n_op_to_remove_per_machine
        if max_start_slice == 0:
            start_slice = 0
        else:
            start_slice = np.random.randint(len(assignment.sol_machines[0]) - n_op_to_remove_per_machine)
        end_slice = start_slice + n_op_to_remove_per_machine
        # print(f"\t random_slice {start_slice}")
        logging.info(f"\t random_slice {start_slice}-{end_slice}")
        for m in assignment.sol_machines.keys():
            assignment.remove_slice(m, start_slice, end_slice)
            
    def random_rectangle(self, assignment:SchedulingAssignment, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # print(f"\t random_rectangle")
        logging.info(f"\t random_rectangle")
        # Destroy op in no more than 3 machines
        machines_to_remove = np.random.choice(
            self.inst.lst_machines,
            3,
            replace=False
        )
        leght_portion = int(self.inst.n_jobs / 3)
        portion_jobs = np.random.randint(0,3)
        # first part, second part, third part
        n_op_to_remove_per_machine = math.ceil(self.OP_TO_DESTROY / 3)
        if n_op_to_remove_per_machine < leght_portion:
            start_slice = leght_portion * portion_jobs + np.random.randint(leght_portion - n_op_to_remove_per_machine)
        else:
            start_slice = leght_portion * portion_jobs
        end_slice = min(start_slice + n_op_to_remove_per_machine, self.inst.n_jobs - 1)
        logging.info(f"\t m{machines_to_remove} from {start_slice} to {end_slice}")
        for m in machines_to_remove:
            assignment.remove_slice(m,start_slice,end_slice)

    def dual_machines(self, assignment:SchedulingAssignment):
        # compute dual costs (add a small value to prevent 0 prob problems)
        potentials = self._compute_dual_sum_machine(assignment)
        # compute machine to destroy
        n_machine_to_destroy = math.ceil(self.OP_TO_DESTROY / self.inst.n_jobs)
        # select them based on potential
        machine_to_destroy = np.random.choice(
            self.inst.n_machines,
            n_machine_to_destroy,
            p=potentials/sum(potentials),
            replace=False
        )
        logging.info(f"\t dual_machines: {machine_to_destroy}")
        # remove machine selected
        for m in machine_to_destroy:
            assignment.remove_machine(m)

    def _compute_dual_sum_machine(self, assignment):
        # run timing
        self.solver_timing._run_optimization(
            assignment
        )
        potentials = 0.0001 + np.zeros((len(assignment.sol_machines)))
        for i, m in enumerate(assignment.sol_machines):
            dual = [ele.getAttr(grb.GRB.Attr.Pi) for _, ele in self.solver_timing.precedence_constraints[m].items()]
            potentials[i] += sum(dual)
        return potentials

    def dual_ops(self, assignment:SchedulingAssignment):
        logging.info(f"\t dual_ops")
        potentials, null_potential = self._compute_dual_ops(assignment)
        potential_ops = list(potentials.keys())
        ops_to_remove = []
        if self.OP_TO_DESTROY > len(potential_ops):
            ops_to_remove.extend(potential_ops)
            rnd_positions = np.random.choice(
                len(null_potential), self.OP_TO_DESTROY - len(potential_ops), replace=False
            )
            for pos in rnd_positions:
                ops_to_remove.append(null_potential[pos])
        else:
            # compute prob according to potential
            probs = np.array(list(potentials.values())) 
            probs /= sum(probs)
            # sample according to potential
            rnd_positions = np.random.choice(
                len(potential_ops), self.OP_TO_DESTROY, p=probs, replace=False
            )
            ops_to_remove = [potential_ops[pos] for pos in rnd_positions]
        for op in ops_to_remove:
            assignment.remove_op(op)

    def _compute_dual_ops(self, assignment):
        # run timing
        self.solver_timing._run_optimization(
            assignment
        )
        # compute dual costs
        potentials = {}
        null_potential = []
        for m in assignment.sol_machines:
            # we have n op in machine - 1 constraints:
            for i, ele in self.solver_timing.precedence_constraints[m].items():
                dual_var = ele.getAttr(grb.GRB.Attr.Pi)
                if dual_var > 0:
                    potentials[assignment.sol_machines[m][i]] = dual_var
                else:
                    null_potential.append(assignment.sol_machines[m][i])
        return potentials, null_potential

    def dual_jobs(self, assignment:SchedulingAssignment):
        n_jobs_to_destroy = math.ceil(self.OP_TO_DESTROY / self.inst.n_machines)
        n_bad_job = int(np.ceil(n_jobs_to_destroy / 2))
        n_good_job = int(np.ceil(n_jobs_to_destroy / 2))
        # compute dual costs
        potentials = self._compute_dual_sum_jobs(assignment)
        prob = potentials/sum(potentials)
        id_bad_job = np.random.choice(
            self.inst.n_jobs, n_bad_job,
            p=prob, replace=False
        )
        id_good_job = np.random.choice(
            self.inst.n_jobs, n_good_job,
            p=(1 - prob)/sum(1-prob), replace=False
        )
        id_job_to_destroy = np.hstack([id_bad_job,id_good_job])
        # id_job_to_destroy = np.random.choice(
        #     self.inst.n_jobs, n_good_job,
        #     p=prob / sum(prob), replace=False
        # )
        logging.info(f"\t dual_jobs: {id_job_to_destroy}")
        self._remove_jobs(assignment, id_job_to_destroy)
    
    def _compute_dual_sum_jobs(self, assignment):
        # run timing
        self.solver_timing._run_optimization(
            assignment
        )
        potentials = np.zeros(self.inst.n_jobs)
        for m in assignment.sol_machines:
            for i, ele in self.solver_timing.precedence_constraints[m].items():
                dual_var = ele.getAttr(grb.GRB.Attr.Pi)
                # self.S[sol_machines[m][i+1]] >= self.C[sol_machines[m][i]]
                idx_job = assignment.sol_machines[m][i][0]
                potentials[idx_job] += dual_var
        return potentials

    def big_setups(self, assignment):
        _, _, setup, _ = self.solver.compute_timing(
            assignment
        )
        setup_times = [ele['t_end'] - ele['t_start'] for ele in setup]
        prob = setup_times/sum(setup_times)
        pos_setup_selected = np.random.choice(len(setup_times), int(self.OP_TO_DESTROY / 2), p=prob, replace=False)
        for pos_setup in pos_setup_selected:
            op_from = self.inst.op_from_j_m[setup[pos_setup]['from'], setup[pos_setup]['machine']]
            
            op_to = self.inst.op_from_j_m[setup[pos_setup]['to'], setup[pos_setup]['machine']]
            
            assignment.remove_op( (op_from, setup[pos_setup]['from']) )
            assignment.remove_op( (op_to, setup[pos_setup]['to']) )

