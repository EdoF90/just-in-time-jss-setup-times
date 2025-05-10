#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
from solvers import *
from data_interfaces import *
from sol_representation import *
from solvers import SolveJITJSSST


np.random.seed(4)
if __name__ == '__main__':
    log_name = os.path.join(".", "logs", "main.log")
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    TIME_LIMIT = 10
    # Instance params
    n_jobs = 10
    n_machines = 10
    early_tardy_tradeoff = 'equal'
    due_date_type = 'loose'
    # Read instance
    instance_name = f"I-{n_jobs}x{n_machines}-{early_tardy_tradeoff}-{due_date_type}-0"
    path_file = os.path.join(
        '.', 'data',
        instance_name
    )
    # read instance
    inst = read_jit_jss_setup_instances(path_file)
    # SOLVE WITH EXACT SOLVER:
    solver = SolveJITJSSST(inst)
    solver.solve(
        verbose=False,
        time_limit=TIME_LIMIT
    )
    _, assignment, _ = solver.get_assignment()
    of_ex, df_sol_ex, setups_ex, info_comp = solver.get_solution()
    print(f"Gurobi of: {of_ex:.2f}")
    # # decoment to plot the solution:
    # plot_gantt_chart(
    #     df_sol_ex,
    #     setups_ex,
    #     inst
    # )
    # SOLVE WITH OR TOOLS
    or_solver = OrToolJITJSSST(inst)
    assignment_or_tool, info_or_tool = or_solver.solve(
        time_limit=TIME_LIMIT
    )
    print(f"Or tool of: {info_or_tool['of_val']:.2f}")
    
    # COMPUTE VNS
    solver = SolveJITJSSST(inst)
    of_vns, df_sol, setups, info_vns = run_vns(
        inst, solver,
        time_limit=TIME_LIMIT,
        verbose=False
    )
    print(f"VNS of: {of_vns:.2f}")
