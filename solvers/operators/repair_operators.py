# -*- coding: utf-8 -*-
import random

def random_repair(state, rnd_state=0):
    
    if len(state.op_removed) == 0:
        return state
    while state.op_removed:
        op_to_set, old_machine = state.op_removed.pop()

        eligible_machines = [ele for ele in state.inst.eligible_machines[op_to_set] if ele != old_machine]
        if len(eligible_machines) == 0:
            eligible_machines = [old_machine]

        new_rnd_machine = random.choice(eligible_machines)
        
        pred = [ele for ele in state.inst.operations_forest.predecessors(op_to_set)]
        min_pos = 0
        if len(pred) != 0:
            machine_pred = [state.assignment.sol_ps[ele] for ele in pred]
            if new_rnd_machine in machine_pred:
                # we have an infeasible solution if we set an operation to start before
                # its predecessor on the same machine
                pos_pred = 0
                for i, op in enumerate(state.assignment.sol_machines[new_rnd_machine]):
                    if op in pred:
                        pos_pred = i
                min_pos = pos_pred + 1

        new_pos = random.randint(min_pos, len(state.assignment.sol_machines[new_rnd_machine]))
        state.assignment.insert_pos(new_pos, new_rnd_machine, op_to_set)    
        print(f'inserting: {op_to_set} to {new_rnd_machine} [pos: {new_pos}]')
    return state

