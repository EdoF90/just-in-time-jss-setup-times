# -*- coding: utf-8 -*-
from instances import *


class Solver():
    """
    General class for all the solvers
    """
    def __init__(self, inst: Instance, settings: dict = {}):
        self.inst = inst
        self.settings = settings
