# -*- coding: utf-8 -*-
import time
import localsolver
from instances import InstanceJobShopSetUp


def solve_with_localsolver(inst: InstanceJobShopSetUp, time_limit: int):
    # Constant for incompatible machines
    INFINITE = 1000000
    
    nb_jobs = inst.n_jobs
    nb_machines = inst.n_machines
    nb_tasks = len(inst.lst_operations)
    
    # number operation for each job
    nb_operations = inst.n_ops
    
    # TODO: rename task in idx_op
    # matrix tasks x machines
    task_processing_time_data = []
    for t in range(nb_tasks):
        task_processing_time_data.append([INFINITE] * nb_tasks)
        op = inst.lst_operations[t]
        for _,row in inst.df_operations[inst.df_operations.op==op].iterrows():
            task_processing_time_data[t][row['machines']] = row['duration_h']
    # precedence: list of lists job x n_operations of the job. each element is a task
    job_operation_task = []
    offset = 0
    for j in range(nb_jobs):
        job_operation_task.append(
            [offset + ele for ele in range(inst.n_ops[j])]
        )
        offset += inst.n_ops[j]

    # Setup: n_machines x op1 x op2
    task_setup_time_data = []
    for m in range(inst.n_machines):
        task_setup_time_data.append([])
        for i in range(nb_tasks):
            task_setup_time_data[m].append([])
            for j in range(nb_tasks):
                op1 = inst.lst_operations[i]
                op2 = inst.lst_operations[j]
                task_setup_time_data[m][i].append(
                    inst.get_setup(m, op1[1], op2[1])
                )
    
    # max starting time
    max_start = 10000

    ls = localsolver.LocalSolver()
    # Declare the optimization model
    model = ls.model

    # MY VARIABLES:
    tardiness = [model.float(0, INFINITE) for _ in range(nb_jobs)]
    earliness = [model.float(0, INFINITE) for _ in range(nb_jobs)]
    flow_time = [model.float(0, INFINITE) for _ in range(nb_jobs)]

    # Sequence of tasks on each machine
    jobs_order = [model.list(nb_tasks) for _ in range(nb_machines)]
    machines = model.array(jobs_order)

    # Each task is scheduled on a machine
    model.constraint(model.partition(machines))

    # Only compatible machines can be selected for a task
    for t in range(nb_tasks):
        for m in range(nb_machines):
            if task_processing_time_data[t][m] == INFINITE:
                model.constraint(model.not_(model.contains(jobs_order[m], t)))

    # For each task, the selected machine
    task_machine = [model.find(machines, t) for t in range(nb_tasks)]

    task_processing_time = model.array(task_processing_time_data)
    task_setup_time = model.array(task_setup_time_data)

    # Integer decisions: start time of each task
    start = [model.float(0, max_start) for _ in range(nb_tasks)]

    # The task duration depends on the selected machine
    duration = [model.at(task_processing_time, t, task_machine[t]) for t in range(nb_tasks)]
    end = [start[t] + duration[t] for t in range(nb_tasks)]

    start_array = model.array(start)
    end_array = model.array(end)

    # Precedence constraints between the operations of a job
    for j in range(nb_jobs):
        for o in range(nb_operations[j] - 1):
            t1 = job_operation_task[j][o]
            t2 = job_operation_task[j][o + 1]
            model.constraint(start[t2] >= end[t1])

    # SETTING RELEASE DATE
    for idx_job, job_info in inst.jobs.items():
        t = inst.lst_operations.index((0,idx_job))
        model.constraint(start[t] >= job_info['release_date'])

    # Setup dalla prima installazione:
    for idx_job, job_info in inst.jobs.items():
        t = inst.lst_operations.index((0,idx_job))
        m = inst.df_operations[inst.df_operations.op==(0,idx_job)].iloc[0].machines
        inst.machines_initial_state[m]
        model.constraint(start[t] >= inst.get_setup(m, inst.machines_initial_state[m], idx_job))

    # Disjunctive resource constraints between the tasks on a machine
    for m in range(nb_machines):
        sequence = jobs_order[m]
        sequence_lambda = model.lambda_function(
            lambda t: start_array[sequence[t + 1]] >= end_array[sequence[t]]
                    + model.at(task_setup_time, m, sequence[t], sequence[t + 1]))
        model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))
    
    for idx_job, job_info in inst.jobs.items():
        t = inst.lst_operations.index((inst.n_ops[idx_job] - 1, idx_job))
        t_0 = inst.lst_operations.index((0, idx_job))
        model.constraint(tardiness[idx_job] >= end[t] - job_info['due_date'] )
        model.constraint(earliness[idx_job] >= job_info['due_date'] - end[t] )
        model.constraint(flow_time[idx_job] == end[t] - start[t_0] )

    of = model.sum(inst.jobs[j]['tardiness_penalty'] * tardiness[j] for j in range(nb_jobs))
    of += model.sum(inst.jobs[j]['earliness_penalty'] * earliness[j] for j in range(nb_jobs))
    of += model.sum(inst.jobs[j]['flow_time_penalty'] * flow_time[j] for j in range(nb_jobs))
    model.minimize(of)

    model.close()

    # Parameterize the solver
    ls.param.time_limit = time_limit
    start = time.time()
    ls.solve()
    end = time.time()
    print(of.value)
    assignment = {}
    for m in range(nb_machines):
        # creation assignment dict
        assignment[m] = [inst.lst_operations[ele] for ele in jobs_order[m].value]
    return assignment, end - start
