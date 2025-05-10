# -*- coding: utf-8 -*-
import time
import collections
from solvers import *
from data_interfaces import *
from sol_representation import *
from ortools.sat.python import cp_model
from ..solution_classes import SchedulingAssignment
from instances.instanceJobShopSetUp import InstanceJobShopSetUp

class OrToolJITJSSST():
    def __init__(self, inst: InstanceJobShopSetUp, settings: dict = {}):
        jobs_data = []
        for j in range(len(inst.jobs)):
            jobs_data.append([0] * inst.n_ops[j])

        for _, row in inst.df_operations.iterrows():
            jobs_data[row['op'][1]][row['op'][0]] = (row['machines'], row['duration_h'])

        machines_count = inst.n_machines
        all_machines = range(machines_count)
        
        # Computes max value for starting and ending times
        max_horizon = sum(task[1] for job in jobs_data for task in job)

        # Create the model.
        model = cp_model.CpModel()
        
        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            'assigned_task_type',
            'start job index duration'
        )

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        # Define variables:    
        earliness_var_lst, tardiness_var_lst, flow_time_var_lst = [], [], []
        for job_id, job in enumerate(jobs_data):
            suffix_job = '_%i' % (job_id)
            # earliness
            earliness_var_lst.append(
                model.NewIntVar(0, max_horizon, 'earliness' + suffix_job)
            )
            # tardiness
            tardiness_var_lst.append(
                model.NewIntVar(0, max_horizon, 'tardiness' + suffix_job)
            )
            # flow time
            flow_time_var_lst.append(
                model.NewIntVar(0, max_horizon, 'flow_time' + suffix_job)
            )
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, max_horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, max_horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(
                    start_var, duration, end_var,
                    'interval' + suffix
                )
                # Duration of the tasks
                all_tasks[job_id, task_id] = task_type(
                    start=start_var,
                    end=end_var,
                    interval=interval_var
                )
                # No setup time setting:
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            job_starts = []
            job_ends = []    
            for j in range(inst.n_jobs):
                for k in range(inst.n_ops[j]):
                    if inst.eligible_machines[k,j][0] == machine:
                        job_starts.append(all_tasks[(j, k)].start)
                        job_ends.append(all_tasks[(j, k)].end)

            model.AddNoOverlap(machine_to_intervals[machine])
            arcs = []
            for j1 in range(len(machine_to_intervals[machine])):
                # Initial arc from the dummy node (0) to a task.
                start_lit = model.NewBoolVar('%i is first job' % j1)
                arcs.append([0, j1 + 1, start_lit])
                # Final arc from an arc to the dummy node.
                arcs.append([j1 + 1, 0, model.NewBoolVar('%i is last job' % j1)])
                for j2 in range(len(machine_to_intervals[machine])):
                    if j1 == j2:
                        continue

                    lit = model.NewBoolVar('%i follows %i' % (j2, j1))
                    arcs.append([j1 + 1, j2 + 1, lit])
                    # We add the reified precedence to link the literal with the
                    # times of the two tasks.
                    model.Add(job_starts[j2] >= job_ends[j1] +
                            inst.get_setup(m=machine, j0=j1, j1=j2)).OnlyEnforceIf(lit)
            model.AddCircuit(arcs)

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id +
                                    1].start >= all_tasks[job_id, task_id].end)

        # Relase date:
        for job_id, job in enumerate(jobs_data):
            model.Add(all_tasks[job_id, 0].start >= int(inst.jobs[job_id]['release_date']))

        # define earliness, tardiness, flowtime
        for job_id, job in enumerate(jobs_data):
            model.Add(
                all_tasks[job_id, inst.n_ops[job_id] - 1].end + earliness_var_lst[job_id] - tardiness_var_lst[job_id] == int(inst.jobs[job_id]['due_date'])
            )
            model.Add(
                flow_time_var_lst[job_id] == all_tasks[job_id, inst.n_ops[job_id] - 1].end - all_tasks[job_id, 0].start
            )
        # Set objective function:       
        model.Minimize(
            sum(
                [inst.jobs[j]['earliness_penalty'] * earliness_var_lst[j]+ inst.jobs[j]['tardiness_penalty'] * tardiness_var_lst[j] + inst.jobs[j]['flow_time_penalty'] * flow_time_var_lst[j] for j in range(inst.n_jobs)]
            )
        )
        self.model = model
        self.inst = inst
        self.jobs_data = jobs_data
        self.assigned_task_type = assigned_task_type
        self.all_tasks = all_tasks
        self.all_machines = all_machines
        self.earliness_var_lst = earliness_var_lst
        self.tardiness_var_lst = tardiness_var_lst
        self.flow_time_var_lst = flow_time_var_lst

    def perturbe_of(self):    
        self.model.Minimize(
            sum(
                [self.inst.jobs[j]['earliness_penalty'] * self.earliness_var_lst[j]+ self.inst.jobs[j]['tardiness_penalty'] * self.tardiness_var_lst[j] for j in range(self.inst.n_jobs)]
            )
        )
        
    def solve(self, time_limit: int, verbose=False):
        # create dictionary to collect info
        info = {}
        # Creates the solver and solve the model
        solver = cp_model.CpSolver()
        # setting time limit
        solver.parameters.max_time_in_seconds = time_limit
        start = time.time()
        # solving the model
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            # get data from assignment
            for job_id, job in enumerate(self.jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        self.assigned_task_type(
                            start=solver.Value(
                                self.all_tasks[job_id, task_id].start
                            ),
                            job=job_id,
                            index=task_id,
                            duration=task[1]
                        )
                    )

            # define best assignment:
            assignment = {m:[] for m in range(self.inst.n_machines)}
            # Create per machine output lines.
            for machine in self.all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                assignment[machine] = [(ele.index, ele.job) for ele in assigned_jobs[machine]]
            
            # Finally print the solution found.
            if verbose:
                print(f'OF or tools: {solver.ObjectiveValue()}')
            info['of_val'] = solver.ObjectiveValue()
        else:
            assignment = {}
        info['comp_time'] = time.time() - start
        return SchedulingAssignment(assignment), info
