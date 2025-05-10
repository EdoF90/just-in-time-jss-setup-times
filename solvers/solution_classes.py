# -*- coding: utf-8 -*-
from sol_representation.plot_gant import plot_gantt_chart


class SchedulingAssignment():
    def __init__(self, sol_assignment: dict):
        # sol_machine: for each machine operations
        self.sol_machines = sol_assignment
        # sol_op: for each operation the machine
        self.sol_op = {}
        for id_machine, lst_op in self.sol_machines.items():
            for op in lst_op:
                self.sol_op[op] = id_machine
        self.removed_ops = []

    def insert_pos(self, operation, machine, pos):
        self.sol_machines[machine].insert(pos, operation)
        self.sol_op[operation] = machine

    def remove_op(self, operation):
        machine = self.sol_op[operation]
        pos_to_remove = self.sol_machines[machine].index(operation)
        del self.sol_machines[machine][pos_to_remove]
        self.sol_op.pop(operation, None)
        # add info to remove_ops
        self.removed_ops.append(operation)
    
    def remove_machine_pos(self, machine, pos):
        op = self.sol_machines[machine][pos]
        del self.sol_machines[machine][pos]
        self.sol_op.pop(op, None)
        # add info to remove_ops
        self.removed_ops.append(op)

    def remove_machine(self, m):
        for op in self.sol_machines[m]:
            self.sol_op.pop(op, None)
            # add info to remove_ops
            self.removed_ops.append(op)
        self.sol_machines[m] = []

    def remove_slice(self, m, start_slice, end_slice):
        # TODO: similar to the method above
        for op in self.sol_machines[m][start_slice:end_slice]:
            self.sol_op.pop(op, None)
            # add info to remove_ops
            self.removed_ops.append(op)
        del self.sol_machines[m][start_slice:end_slice]
    
    def move_op(self, operation, new_machine, pos=0):
        # TODO: this is an operator, move in the right file
        new_assignment = self.__copy__()
        old_machine = new_assignment.sol_op[operation]
        new_assignment.sol_machines[old_machine].remove(operation)
        new_assignment.insert_pos(operation, new_machine, pos)
        return new_assignment

    def move_op_within(self, operation, pos):
        machine = self.sol_op[operation]
        self.sol_machines[machine].remove(operation)
        self.sol_machines[machine].insert(pos, operation)

    def swap_ops(self, op1, op2):
        new_assignment = self.__copy__()
        # get useful data
        machine1 = new_assignment.sol_op[op1]
        machine2 = new_assignment.sol_op[op2]
        pos1 = new_assignment.sol_machines[machine1].index(op1)
        pos2 = new_assignment.sol_machines[machine2].index(op2)
        # switch machine in sol_op
        new_assignment.sol_op[op1] = machine2
        new_assignment.sol_op[op2] = machine1
        # switch pos and order in sol_machine
        new_assignment.sol_machines[machine1][pos1] = op2
        new_assignment.sol_machines[machine2][pos2] = op1
        return new_assignment

    def __str__(self) -> str:
        return f"{self.sol_machines}"
    
    def __repr__(self):
        return f"{self.sol_machines}"
    
    def __copy__(self):
        tmp = {}
        for key, ele in self.sol_machines.items():
            tmp[key] = [el for el in ele]
        copied = SchedulingAssignment(
            tmp
        )
        copied.removed_ops = [ ele for ele in self.removed_ops]
        return copied 

class SchedulingSolution():
    def __init__(
        self, assignment, inst
    ):
        super().__init__()
        self.assignment = assignment
        self.inst = inst
        self.of = None
        self.df_sol = None
        self.setups = None

    def move_op(self, operation, new_machine, pos=0):
        # TODO: this is an operator, move in the right file
        new_assignment = self.assignment.move_op(
            operation=operation,
            new_machine=new_machine,
            pos=pos
        )
        self.of, self.df_sol, self.setups, self.comp_time = self.solver.compute_timing(
            new_assignment
        )

    def plot(self):
        plot_gantt_chart(
            self.sol,
            self.setups,
            self.inst
        )
