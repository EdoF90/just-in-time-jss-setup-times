# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd


def create_jit_jss_setup_instance(data, folder_to_write):
    '''
    Instance generation following Baptiste Flamini 2008
    '''
    if not os.path.exists(folder_to_write):
        os.mkdir(folder_to_write)
    else:
        print('Folder already existing')
        return
    n_jobs = data['n_jobs']
    n_machines = data['n_machines']
    # ################
    # OPERATIONS
    # ################
    outfile = open(os.path.join(folder_to_write, "operations.csv"), "w")
    
    lst_machines = list(range(n_machines))
    job_duration = []
    job_goes_in_machines = []
    machine_job = [[] for x in range(n_machines)]
    for i in range(n_jobs):
        # define the processing times
        processing_times = np.random.randint(
            low=10,
            high=30,
            size=(n_machines)
        )
        job_duration.append(
            sum(processing_times)
        )
        # shuffle the machines
        random.shuffle(lst_machines)
        new_machine_lst = [lst_machines[0]]
        for idx_machine in range(1, len(lst_machines)):
            if np.random.binomial(1, 0.9) == 1:
                new_machine_lst.append(lst_machines[idx_machine])
        job_goes_in_machines.append([ele for ele in new_machine_lst])
        for idx_machine in new_machine_lst:
            machine_job[idx_machine].append(i)
        outfile.write(
            ','.join([f"{new_machine_lst[i]},{processing_times[i]}" for i in range(len(new_machine_lst))])
        )
        outfile.write("\n")
    outfile.close()
    # ################
    # JOBS
    # ################
    data_job = {
        'id_job':[],
        'release_date':[],
        'due_date':[],
        'earliness_penalty':[],
        'tardiness_penalty':[],
        'flow_time_penalty':[],
    }
    for i in range(n_jobs):
        data_job['id_job'].append(i)
        release_date = np.random.randint(0, 101)
        data_job['release_date'].append(
            release_date
        )
        if data['due_date_type'] == 'tight':
            due_date = release_date + 1.3 * job_duration[i] 
        elif data['due_date_type'] == 'loose':
            due_date = release_date + 1.5 * job_duration[i] 
        elif data['due_date_type'] == 'extra_loose':
            due_date = release_date + 1.7 * job_duration[i] 
        elif data['due_date_type'] == 'extra_tight':
            due_date = release_date + 1.1 * job_duration[i] 
        else:
            raise ValueError("due_date_type entered not valid")

        data_job['due_date'].append(
            due_date
        )
        penalty = np.round(
            np.random.uniform(0.1, 1),
            decimals=2
        )
        if data['early_tardy_tradeoff'] == 'equal':
            penalty_early = penalty
            penalty_tardy = penalty
        else:
            penalty_early = max(np.round(
                np.random.uniform(0.1, 0.3),
                decimals=2
            ), penalty)
            penalty_tardy = penalty

        data_job['earliness_penalty'].append(
            penalty_early
        )
        data_job['tardiness_penalty'].append(
            penalty_tardy
        )
        data_job['flow_time_penalty'].append(
            penalty_early * 0.1
        )
    df_jobs = pd.DataFrame.from_dict(data_job)
    df_jobs.to_csv(
        os.path.join(folder_to_write, "jobs.csv"),
        index=False
    )
    # ################
    # SET UP
    # ################
    data_setup ={
        'machine': [],
        'id_job0': [],
        'id_job1': [],
        'time': [],
    }
    for m in range(n_machines):
        for i in range(n_jobs):
            for j in range(n_jobs):
                if (m in job_goes_in_machines[i]) and (m in job_goes_in_machines[j]):
                    # setup_time = np.random.randint(0, 3)
                    setup_time = np.random.choice([0, 5, 10, 30], 1, p=[0.5, 0.3, 0.15, 0.05])[0]
                    if setup_time > 0:
                        data_setup['machine'].append(m)
                        data_setup['id_job0'].append(i)
                        data_setup['id_job1'].append(j)
                        data_setup['time'].append(
                            setup_time
                        )
    df_setup = pd.DataFrame.from_dict(data_setup)
    df_setup.to_csv(
        os.path.join(folder_to_write, "setup.csv"),
        index=False
    )
    # ################
    # SETTINGS
    # ################
    settings_data = {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
    }
    fp = open(
        os.path.join(
            folder_to_write,
            "settings.json"
        ),
        "w"
    )
    json.dump(settings_data, fp, indent=4)
    fp.close()
    # ################
    # INITIAL STATE
    # ################
    outfile = open(os.path.join(folder_to_write, "initial_setup.csv"), "w")
    outfile.write("machine, job\n")
    for m in range(n_machines):
        outfile.write(
            f"{m},{random.choice(machine_job[m])}\n"
        )
    outfile.close()
