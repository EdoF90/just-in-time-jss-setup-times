# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
from .instance import Instance
import matplotlib.pyplot as plt


class InstanceJobShop(Instance):
    def __init__(self, n_jobs, n_machines, lst_job, due_dates):
        super(InstanceJobShop, self).__init__(n_jobs, n_machines)
        self.lst_machines = range(n_machines)
        self.n_operations = n_jobs * n_machines
        self.lst_job = lst_job
        # Data structures for gantt plot
        operations = []
        self.jobs = {}
        for j, job in enumerate(lst_job):
            self.jobs[j] = {
                'item_name': j,
                'due_date': due_dates[j],
            }
            for pos, op in enumerate(job):
                operations.append(
                    {
                        'op': (pos, j),
                        'order': j,
                        'id_order': f"ord{j}",
                        'machines': op['machine'],
                        'duration_h': op['processing_time'],
                    }
                )
        self.df_operations = pd.DataFrame.from_dict(operations)


