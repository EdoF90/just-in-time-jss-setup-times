# -*- coding: utf-8 -*-
import time
import logging
import numpy as np
from solvers import *
from sol_representation import *
from .exact_model import SolveJITJSSST
from .solve_or_tool import OrToolJITJSSST
from .operators_destroy import DestroyToolkit
from .operators_repair import RepairToolkit
from ..solution_classes import SchedulingAssignment


def __define_settings(inst):
    if inst.n_jobs == 5:
        # Optimal setting for 5 jobs
        OP_TO_DESTROY = inst.n_machines * 3
        destroyers = DestroyToolkit(inst, OP_TO_DESTROY)
        time_limit = 2
        MAX_SOL_TIME = 2
        destroyers_lst = [
            {"fnc": destroyers.random_machines, "delta_f": [], "name": "random_machine"},
            {"fnc": destroyers.dual_ops, "delta_f": [], "name": "dual_ops"},
            {"fnc": destroyers.random_slice, "delta_f": [], "name": "random_slice"},
        ]
        solver_time_increment = 0
        gap_limit = 0.1

    elif inst.n_jobs == 10:
        # Optimal setting for 10 jobs
        OP_TO_DESTROY = inst.n_machines * 3
        destroyers = DestroyToolkit(inst, OP_TO_DESTROY)
        time_limit = 10
        MAX_SOL_TIME = 30
        destroyers_lst = [
            {"fnc": destroyers.dual_ops, "delta_f": [], "name": "dual_ops"},
            {"fnc": destroyers.random_machines, "delta_f": [], "name": "random_machines"},
            {"fnc": destroyers.random_slice, "delta_f": [], "name": "random_slice"},
        ]
        solver_time_increment = 0
        gap_limit = 0.2

    elif inst.n_jobs == 20:
        # Optimal setting for 20 jobs
        time_limit = 15
        OP_TO_DESTROY = inst.n_machines
        destroyers = DestroyToolkit(inst, OP_TO_DESTROY)
        MAX_SOL_TIME = 30
        destroyers_lst = [
            {"fnc": destroyers.dual_ops, "delta_f": [], "name": "dual_ops"},
            {"fnc": destroyers.random_slice, "delta_f": [], "name": "random_slice"},
            {"fnc": destroyers.dual_machines, "delta_f": [], "name": "dual_machines"},
            {"fnc": destroyers.worst_jobs, "delta_f": [], "name": "worst_jobs"},
            {"fnc": destroyers.dual_jobs, "delta_f": [], "name": "dual_jobs"}
        ]
        solver_time_increment = 5
        gap_limit = 0.2
    elif inst.n_jobs == 50:
        # Optimal setting for 50 jobs
        time_limit = 30
        OP_TO_DESTROY = inst.n_machines * 2
        destroyers = DestroyToolkit(inst, OP_TO_DESTROY)
        MAX_SOL_TIME = 30
        destroyers_lst = [
            {"fnc": destroyers.dual_ops, "delta_f": [], "name": "dual_ops"},
            {"fnc": destroyers.random_slice, "delta_f": [], "name": "random_slice"},
            {"fnc": destroyers.worst_jobs, "delta_f": [], "name": "worst_jobs"},
            {"fnc": destroyers.dual_jobs, "delta_f": [], "name": "dual_jobs"}
        ]
        solver_time_increment = 0
        gap_limit = 0.2
    return destroyers, time_limit, OP_TO_DESTROY, MAX_SOL_TIME, destroyers_lst, solver_time_increment, gap_limit

# Reduced Variable Neiborhood Search
def run_vns(inst, solver: SolveJITJSSST, time_limit, verbose=False):
    logging.info("# ################## #")
    logging.info("# START VNS #")
    logging.info("# ################## #")
    destroyers, solver_time_limit, OP_TO_DESTROY, MAX_SOL_TIME, destroyers_lst, solver_time_increment, gap_limit = __define_settings(inst)    
    repairers = RepairToolkit(inst, solver, verbose)
    start = time.time()
    # initialize time to best
    time_to_best = 0
    # if too many operation to destroy, solve exact problem
    if OP_TO_DESTROY >= sum(inst.n_ops):
        if verbose:
            print("Too many ops to destroy, solving exact model")
        solver._remove_assignment()
        solver.solve()
        of, df_sol, setups, _ = solver.get_solution()
        stop = time.time()
        comp_time = stop - start
        info = {
            "comp_time": comp_time,
            "time_to_best": comp_time,
            "prob_destroyers": [],
            "prob_repairers": [],
        }
        return of, df_sol, setups, info
    
    precedence_rules_vns = ['EDD', 'RND']
    # INITIALIZATION:
    of_best = np.inf
    new_sol_counter = 0
    if inst.n_jobs in [10, 20]:
        or_solver = OrToolJITJSSST(inst)

    while time.time() - start < time_limit:
        rule_idx = min(new_sol_counter, len(precedence_rules_vns)-1)
        # the last rule is random, hence it will provide different solutions each time
        if inst.n_jobs < 10 or inst.n_jobs == 50:
            assignment = list_scheduling(
                inst,
                rule=precedence_rules_vns[rule_idx]
            )
        else:
            or_solver.perturbe_of()
            assignment, _ = or_solver.solve(
                time_limit=10
            )
        new_sol_counter += 1
        # compute first solution
        old_of, df_assignment, _, _ = solver.compute_timing(
            assignment
        )
        logging.info(f"[{(time.time() - start):.0f}] - {precedence_rules_vns[rule_idx]} >>> STARTING OF: {old_of}")
        
        # Resetting parameters:
        tmp_solver_time_limit = solver_time_limit
        tmp_op_to_destroy = OP_TO_DESTROY
        idx_destroyer = 0
        # Start loop for improvement
        while idx_destroyer < len(destroyers_lst):
            # update operation to destroy
            destroyers.set_n_op_to_destroy(tmp_op_to_destroy)
            # create a copy of assignment.sol_machines
            new_assignment = assignment.__copy__()
            # copy the solution before the destroy (it is a useful starting point)
            pre_destroy_sol = {}
            for key, lst_ops in assignment.sol_machines.items():
                pre_destroy_sol[key] = []
                for ops in lst_ops:
                    pre_destroy_sol[key].append(ops)
            # apply destroy
            destroyers_lst[idx_destroyer]['fnc'](new_assignment)
            # apply repair
            new_of, _, info = repairers.optimal_allocation(
                new_assignment,
                pre_destroy_sol,
                time_limit=tmp_solver_time_limit,
                gap=gap_limit,
                verbose=False
            )
            destroyers_lst[idx_destroyer]['delta_f'].append(
                (old_of - new_of) / old_of
            )
            logging.info(f"\t[N] MIPgap: {info['mip_gap']:.2f}, t:{info['tardiness']:.2f}, e:{info['earliness']:.2f}, f:{info['flow_time']:.2f}")
            if new_of < old_of:
                # update the value if improve
                logging.info(f"Found new obj func: {new_of} [ from N{idx_destroyer}] ")
                idx_destroyer = 0
                # if the difference is low:
                if old_of - 0.5 < new_of:
                    # increase the solver time
                    tmp_solver_time_limit += min(tmp_solver_time_limit + solver_time_increment, MAX_SOL_TIME)
                    logging.info(f">> UPDATE: {tmp_op_to_destroy}op - {tmp_solver_time_limit}s")
                assignment = new_assignment
                old_of = new_of
            else:
                # otherwise chance neiborhood
                logging.info(">>>>>> STUCK <<<<<<")
                idx_destroyer += 1
                tmp_solver_time_limit = min(tmp_solver_time_limit + solver_time_increment, MAX_SOL_TIME)
                logging.info(f"\t Change neiborhood, new op_to_destroy: {tmp_op_to_destroy}op - {tmp_solver_time_limit}s")
            # if better than the best save solution
            if new_of < of_best:
                time_to_best = time.time() - start
                # Updating the assignment in a temporary dict
                best_assignment = {}
                for key, lst_ops in new_assignment.sol_machines.items():
                    best_assignment[key] = []
                    for ops in lst_ops:
                        best_assignment[key].append(ops)
                # updating the best value
                of_best = new_of

            # Check time inner while
            if time.time() - start > time_limit:
                break

    # return the best solution:
    solver.set_assignment(SchedulingAssignment(best_assignment))
    solver.solve(verbose=False)# , lp_name="final_timing")
    of, df_sol, setups, _ = solver.get_solution()
    stop = time.time()
    comp_time = stop - start
    info = {
        "comp_time": comp_time,
        "time_to_best": time_to_best
    }
    for ele in destroyers_lst:
        logging.info(f"{ele['name']}: {ele['delta_f']}")
    return of, df_sol, setups, info
