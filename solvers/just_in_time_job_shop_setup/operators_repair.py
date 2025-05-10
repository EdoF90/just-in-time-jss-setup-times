# -*- coding: utf-8 -*-
import logging
import numpy as np
from solvers import *
from instances import Instance
from .solveJITJSSSTNeighborhood import SolveJITJSSSTNeighborhood
from solvers.just_in_time_job_shop_setup.exact_model import SolveJITJSSST

class RepairToolkit():
    def __init__(self, inst: Instance, solver: SolveJITJSSST, solver_verbose=False):
        self.inst = inst
        self.solver = solver
        self.solver_verbose = solver_verbose
        self.neighborhood_solver = SolveJITJSSSTNeighborhood(inst)
        

    def exact_repair(self, assignment, pre_destroy_sol:dict, lp_name=None, gap=None, time_limit=None, verbose=False):
        logging.info("exact_repair")
        self.solver.set_destroyed_assignment(
            assignment, pre_destroy_sol
        )
        self.solver.solve(verbose=verbose, lp_name=lp_name, gap=gap, time_limit=time_limit)
        of_new, df_sol, _, _ = self.solver.get_solution()
        self.solver._remove_assignment()
        return of_new, df_sol
    
    def optimal_allocation(self, assignment, pre_destroy_sol=None, lp_name=None, gap=None, time_limit=None, verbose=None):
        logging.info("optimal_allocation")
        # allocate the operations removed 
        
        of_new, df_sol, info = self.neighborhood_solver.allocate(
            new_assignment=assignment,
            pre_destroy_sol=pre_destroy_sol,
            verbose=verbose if verbose is not None else self.solver_verbose,
            lp_name=lp_name,
            gap=gap, time_limit=time_limit
        )
        assignment.removed_ops = []
        return of_new, df_sol, info
        
    def repair_with_rule(self, incomplete_assignment, precedence_rule):
        logging.info(f"repair_with_rule {precedence_rule}")
        new_assignment = list_scheduling(self.inst, precedence_rule, incomplete_assignment)
        return new_assignment

    def repair_with_ATCS(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "ATCS")

    def repair_with_EDD(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "EDD")

    def repair_with_LPT(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "LPT")

    def repair_with_SPT(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "SPT")

    def repair_with_MSF(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "MSF")

    def repair_with_WDPTF(self, machine_assignment):
        return self.repair_with_rule(machine_assignment, "WDPTF")
