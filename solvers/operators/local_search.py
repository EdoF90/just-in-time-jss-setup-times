# -*- coding: utf-8 -*-
from pandas import DataFrame
from solvers.abs_solver import Solver
from ..solution_classes import SchedulingAssignment
from instances import Instance
from sol_representation import *


def explore_neighborhood_swap(inst: Instance, solver: Solver, of_starting: float, starting_assignment: SchedulingAssignment):
    # Initialization
    of_best = of_starting
    assignment_best = starting_assignment
    df_sol_best, setups_best = None, None
    # for each operation in the solution
    for machine, ordered_ops in starting_assignment.sol_machines.items():
        for op1 in ordered_ops:
            # consider all the operation that can be done in the machine
            for op2 in inst.machines_ops[machine]:
                # if they are different and they can be swapped:
                new_machine_for_op1 = starting_assignment.sol_op[op2]
                if op1 != op2 and op1 in inst.machines_ops[new_machine_for_op1]:
                    # swap operations
                    new_assignment = starting_assignment.swap_ops(op1, op2)
                    of_new, df_sol_new, setups_new, _ = solver.compute_timing(
                        new_assignment
                    )
                    # if the result is good, update the values
                    if of_new < of_best:
                        of_best = of_new
                        assignment_best = new_assignment
                        df_sol_best = df_sol_new
                        setups_best = setups_new
    return of_best, assignment_best, df_sol_best, setups_best


def explore_neighborhood_relocation(inst: Instance, timing_solver: Solver, neighborhood_solver: Solver, of_starting: float, starting_assignment: SchedulingAssignment, df_sol_best: DataFrame, first_improvement: bool):
    # Initialization
    of_best = of_starting
    assignment_best = starting_assignment
    setups_best = None
    # sort operation with respect to earliness - tardiness
    df_sol_best.sort_values(['tardiness', 'earliness'], ascending=[False, False], inplace=True )
    # For each operation
    for op in df_sol_best.op:
        # set the operation in the best possible position (testing all possible or using neighborhood_solver)
        assignment_new, of_new = best_pos_op(
            inst,
            timing_solver,
            neighborhood_solver,
            of_starting,
            starting_assignment,
            op,
            first_improvement=first_improvement,
        )
        # if it improves the solution update the best
        if of_new < of_best:
            of_best = of_new
            assignment_best = assignment_new
            # compute df_sol e setup_best
            timing_solver._run_optimization(
                assignment_best
            )
            df_sol_best, setups_best = timing_solver._get_solution(
                assignment_best.sol_machines
            )
            # if first_improvement exit as soon as a better solution is met
            if first_improvement:
                return assignment_best, of_best, df_sol_best, setups_best
    return assignment_best, of_best, df_sol_best, setups_best

def best_pos_op(inst: Instance, solver: Solver, neighborhood_solver: Solver, of_best: float, starting_assignment: SchedulingAssignment, operation: tuple, first_improvement: bool):

    if neighborhood_solver:
        new_sol_machine = {key: [] for key in starting_assignment.sol_machines}
        for key in starting_assignment.sol_machines:
            new_sol_machine[key] = [ele for ele in starting_assignment.sol_machines[key] if ele != operation]

        of_new, df_sol, setups = neighborhood_solver.allocate(
            operation,
            starting_assignment,
            new_sol_machine
        )
        print(of_new)
        quit()
    else:       
        old_machine = starting_assignment.sol_op[operation]
        # initialization:
        best_pos = starting_assignment.sol_machines[old_machine].index(operation)
        best_machine = old_machine
        # removing operation from position
        del starting_assignment.sol_machines[old_machine][best_pos]   
        # for all the machines, including the actual one
        for new_machine in inst.eligible_machines[operation]:
            # move op in all possible positions
            for pos in range(len(starting_assignment.sol_machines[new_machine])):
                # moving operation in new_sol_machine
                starting_assignment.sol_machines[new_machine].insert(pos, operation)
                # compute timing
                of_new, _ = solver._run_optimization(
                    starting_assignment
                )
                # if the new sol improves the best, update.
                if of_new < of_best:
                    best_pos = pos
                    best_machine = new_machine
                    of_best = of_new
                    if first_improvement:
                        # update sol op
                        starting_assignment.sol_op[operation] = best_machine
                        return starting_assignment, of_best
                # restoring old solution:
                del starting_assignment.sol_machines[new_machine][pos]
    # updating data
    starting_assignment.sol_machines[best_machine].insert(best_pos, operation)
    starting_assignment.sol_op[operation] = best_machine
    return starting_assignment, of_best

def move_op_best_pos_in_machine(of: float, assignment: SchedulingAssignment, solver: Solver, op: str, new_machine: str):
    of_best = of
    df_sol_best = None
    setups_best = None

    for i in range(len(assignment.sol_machines[new_machine]) + 1):
        new_assignment = assignment.move_op(
            operation=op,
            new_machine=new_machine,
            pos=i
        )
        of_new, df_sol_new, setups_new, _ = solver.compute_timing(
            new_assignment
        )
        if of_new < of_best:
            of_best = of_new
            assignment_best = new_assignment
            df_sol_best = df_sol_new
            setups_best = setups_new
    return of_best, assignment_best, df_sol_best, setups_best


def improve_latest_order(inst, df_sol, assignment, solver, of):
    # get late order
    tardy_final_ops = df_sol[df_sol['tardiness'] > 0]
    order_late = tardy_final_ops.loc[tardy_final_ops.tardiness.idxmax()].order
    # for each component
    for _, row in df_sol[df_sol.order == order_late].iterrows():
        tmp = inst.df_operations[inst.df_operations.op == row.op]
        tmp_other_machine = tmp[tmp.machines != row.machine]
        # if there is a faster machine to execute the operation
        if float(tmp[tmp.machines == row.machine].duration_h) > tmp_other_machine.duration_h.min():
            # move the operation in the best position of that machine
            new_machine = tmp_other_machine.loc[tmp_other_machine.duration_h.idxmin(
            )].machines
            of_best, assignment_best, df_sol_best, setups_best = move_op_best_pos_in_machine(
                of, assignment, solver, row.op, new_machine)
    return of_best, assignment_best, df_sol_best, setups_best
