# -*- coding: utf-8 -*-
import pickle
import numpy as np
import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt


class Instance():
    def __init__(self, n_jobs, n_machines):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.lst_machines = range(n_machines)

    def dump_pickle_file(self, file_path):
        outfile = open(file_path, "wb")
        pickle.dump(self, outfile)
        outfile.close()
