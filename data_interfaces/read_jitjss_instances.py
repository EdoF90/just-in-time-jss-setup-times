# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from instances import *


def read_jit_jss_setup_instances(path_file: str) -> InstanceJobShopSetUp:
    # READ GENERAL INFO
    fp_setting = open(
        os.path.join(
            path_file,  
            "settings.json"
        ), 'r'
    )
    settings = json.load(fp_setting)
    fp_setting.close()
    n_jobs = settings['n_jobs']
    n_machines = settings['n_machines']
    # READ JOBS INFO
    df_jobs = pd.read_csv(
        os.path.join(
            path_file,  
            "jobs.csv"
        )
    )
    # READ SETUP INFO
    df_setup = pd.read_csv(
        os.path.join(
            path_file,  
            "setup.csv"
        )
    )
    fp_operations = open(
        os.path.join(
            path_file,  
            "operations.csv"
        ), 'r'
    )
    rows = fp_operations.readlines()
    lst_job = [ [] for _ in range(n_jobs)]
    # for each line
    for idx_job, row in enumerate(rows):
        row_split = row.strip().split(',')
        for i in range(0, len(row_split), 2):
            lst_job[idx_job].append(
                {    
                    "machine": int(row_split[i]),
                    "processing_time": int(row_split[i + 1])
                }
            )
    fp_operations.close()
    
    fp_initial_setup = open(
        os.path.join(
            path_file,  
            "initial_setup.csv"
        ), 'r'
    )
    machines_initial_state = []
    rows = fp_initial_setup.readlines()
    for i, row in enumerate(rows):
        if i == 0:
            continue
        else:
            machine_state, state = row.strip().split(',')
            machines_initial_state.append(int(state))

    fp_initial_setup.close()
    return InstanceJobShopSetUp(n_jobs, n_machines, machines_initial_state, df_jobs, df_setup, lst_job)

def read_jitjss_instances(path_file: str) -> InstanceJobShop:
    lst_job = []
    job_due_dates = []
    with open(path_file) as f:
        rows = f.readlines()
        # for each line
        for i, row in enumerate(rows):
            # if first line
            if i==0:
                row_split = row.split(' ')
                n_jobs = int(row_split[0])
                n_machines = int(row_split[1])
            else:
                # splitting the row
                lst_jobs_info = row.strip().split('\t\t')
                # create the job
                lst_job.append([])
                for _ in range(n_machines):
                    lst_job[-1].append([])
                # create release date and due dates
                job_due_dates.append(0)
                sum_processing = 0
                for pos, operation in enumerate(lst_jobs_info):
                    tmp = operation.split(' ')
                    # machine to process the task
                    machine = int(tmp[0])
                    # processing time of the task
                    processing_time = float(tmp[1])
                    # duedate of the task
                    due_date = float(tmp[2])
                    # earliness cost
                    earliness_cost = float(tmp[3])
                    # tardiness cost. 
                    tardiness_cost = float(tmp[4])
                    # updating lst_job
                    lst_job[-1][pos] = {
                        "machine": machine,
                        "processing_time": processing_time,
                        "due_date": due_date,
                        "release_date": sum_processing,
                        "earliness_cost": earliness_cost,
                        "tardiness_cost": tardiness_cost,
                    }
                    # update sum processing
                    sum_processing += processing_time
                job_due_dates[-1] = due_date
    return InstanceJobShop(
        n_jobs, n_machines, lst_job, job_due_dates
    )
