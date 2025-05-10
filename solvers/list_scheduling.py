# -*- coding: utf-8 -*-
import random
import numpy as np
from instances import *
from .solution_classes import SchedulingAssignment

precedence_rules = ["EDD", "ATCS", "LPT", "SPT", "MSF", "WDPTF", "RND"]

def list_scheduling(inst: Instance, rule: str = "ATCS", incomplete_assignment: SchedulingAssignment = None) -> SchedulingAssignment:
    if (incomplete_assignment is not None) and isinstance(inst, InstanceJobShopSetUp):
        precedence_op = {}
        for _, ele in incomplete_assignment.sol_machines.items():
            for i in range(len(ele) - 1):
                precedence_op[ele[i+1]] = ele[i]
    ready_ops = []
    ended_ops = []
    sol = {machine: [] for machine in inst.lst_machines}
    available_times_machine = {key: 0 for key in inst.lst_machines}
    # READY OPS INITIALIZATION
    ready_ops.extend(inst.starting_ops)
    t = 0
    while len(ready_ops) > 0:
        # get info for operations that are ready (ready_ops)
        df_ready_ops = inst.df_operations[
            inst.df_operations['op'].isin(ready_ops)
        ]
        useful_machine = df_ready_ops.machines.unique()
        # get available machine
        tmp_dict_availability = {
            machine: time_available for machine,
                    time_available in available_times_machine.items() if machine in useful_machine
        }
        free_machine = min(
            tmp_dict_availability, key=tmp_dict_availability.get
        )
        t = available_times_machine[free_machine]
        # get the best operation for that machine
        if (incomplete_assignment is not None) and isinstance(inst, InstanceJobShopSetUp):
            # if not flexible, then we mantain the same order of partial solution.
            best_op, duration = _get_best_op(
                t, free_machine, sol, df_ready_ops, inst, rule, incomplete_assignment
            )
            # if best_op has a predecessors not yet in the solution:
            if (best_op in precedence_op) and (precedence_op[best_op] not in sol[free_machine]):
                # if the precedence op of one operation is not in sol
                # we add the closest op to the sol
                tentative_op = best_op
                # if tentative_op has a precedence_op not in the sol:
                while (tentative_op in precedence_op) and (precedence_op[tentative_op] not in sol[free_machine]):
                    tentative_op = precedence_op[tentative_op]
                # if the tentative_op is not ready, add the previous op
                if tentative_op not in list(df_ready_ops.op):
                    # while there is no a ready ope
                    while tentative_op not in list(df_ready_ops.op):
                        # define pre_op
                        tentative_op = (tentative_op[0] - 1, tentative_op[1])
                    # get machine of pre_op
                    free_machine = inst.df_operations[inst.df_operations['op'] == tentative_op].iloc[0].machines
                best_op = tentative_op
                duration = inst.df_operations[inst.df_operations.op == best_op].iloc[0].duration_h
        else:
            best_op, duration = _get_best_op(
                t, free_machine, sol, df_ready_ops, inst, rule, incomplete_assignment
            )

        if best_op != "n.d.":
            sol[free_machine].append(
                best_op
            )
            ended_ops.append(best_op)
            _update(ready_ops, ended_ops, best_op, inst.operations_forest)
            available_times_machine[free_machine] = t + duration
    return SchedulingAssignment(sol)


def _get_best_op(t, machine, sol, df_ready_ops, inst, kpi_type="ATCS", incomplete_assignment: SchedulingAssignment = None):
    df_tmp = df_ready_ops[df_ready_ops['machines'] == machine]
    if incomplete_assignment and (not isinstance(inst, InstanceJobShopSetUp)):
        for op in df_tmp.op:
            # if the considered operation is in the incomplete assignment of that machine
            if op in incomplete_assignment.sol_op.keys():
                # we return it
                best_op = op
                duration = df_ready_ops[df_ready_ops['op']
                                        == best_op].iloc[0].duration_h
                return best_op, duration
    best_op = None
    if kpi_type == 'RND':
        kpi_type = random.choice(["EDD", "SPT"])
    # Apparent Tardiness Cost with Setup
    if kpi_type == 'ATCS':
        # get operations ready to be done:
        raw_op = [ele[0] for ele in df_tmp.op]
        # setting parameter k1 and k2:
        k1 = 1
        k2 = 1
        # get weights
        w = df_tmp.importance.values
        # get earliest starting time:
        d = df_tmp.earliest_starting
        # get processing time:
        p = df_tmp.duration_h.values
        if len(sol[machine]) == 0:
            # if it is the first op on the machine we consider the initial setup
            df_setup_tmp = inst.df_setup[(inst.df_setup.machine == machine) & (
                inst.df_setup.op1 == inst.machines_initial_state[machine]) & (inst.df_setup.op2.isin(raw_op))]
        else:
            # otherwise the precedent operation on that machine
            last_op = sol[machine][-1][0]
            df_setup_tmp = inst.df_setup[(inst.df_setup.machine == machine) & (
                inst.df_setup.op1 == last_op) & (inst.df_setup.op2.isin(raw_op))]
        if len(df_setup_tmp) == 0:
            # no setup time:
            s = 0
        else:
            # else, we order and we take the right one.
            s = np.zeros(len(df_tmp))
            for i, ele in enumerate(df_tmp.op):
                s[i] = df_setup_tmp[df_setup_tmp.op2 == ele[0]].time_h if len(
                    df_setup_tmp[df_setup_tmp.op2 == ele[0]]) > 0 else 0
        s_average = float(
            inst.df_setup[inst.df_setup['machine'] == machine][['time_h']].mean())
        kpi = w / p
        kpi *= np.exp(-np.maximum(d - (t + p), 0) / (k1 * np.average(p)))
        kpi *= np.exp(- s / (s_average * k2))
    # Earliest Due Date
    elif kpi_type == 'EDD':  # STATIC
        kpi = (df_tmp.job_due_date).values
    # Longest Processing time first
    elif kpi_type == 'LPT':  # STATIC
        kpi = -(df_tmp.duration_h).values
    # Shortest Processing time first
    elif kpi_type == 'SPT':  # STATIC
        kpi = (df_tmp.duration_h).values
    # Minimum slack first
    elif kpi_type == 'MSF':  # DYNAMIC
        kpi = (df_tmp.earliest_starting).values - \
            (t + (df_tmp.duration_h).values)
    # Weighted Shortest Processing Time first
    elif kpi_type == 'WDPTF':  # STATIC
        kpi = df_tmp.importance.values / df_tmp.duration_h.values
    post_best_op = np.argmin(kpi)
    row_best_op = df_tmp.iloc[post_best_op]
    best_op = row_best_op.op
    duration = row_best_op.duration_h
    return best_op, duration


def _update(ready_ops, ended_ops, ele, operations_forest):
    # COMPUTE SUCCESSORS
    successors = operations_forest.successors(ele)
    # REMOVE ELEMENT
    if ele in ready_ops:
        ready_ops.remove(ele)
    # ADD SUCCESSORS IF NOT ALREADY IN
    for ele in successors:
        all_parents_completed = True
        for pred in operations_forest.predecessors(ele):
            if pred not in ended_ops:
                all_parents_completed = False
                continue
        if (ele not in ended_ops) and (ele not in ready_ops) and all_parents_completed:
            ready_ops.append(ele)
