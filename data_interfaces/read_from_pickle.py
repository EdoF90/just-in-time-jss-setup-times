# -*- coding: utf-8 -*-
import pickle


def read_from_pickle(file_path):
    infile = open(file_path, "rb")
    inst = pickle.load(infile)
    infile.close()
    return inst
