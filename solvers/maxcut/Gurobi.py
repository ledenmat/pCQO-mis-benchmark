import numpy as np
import gurobipy as gb

from lib.Solver import Solver

import networkx as nx
from networkx import Graph

# PARAMS:


class Gurobi(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()

        self.time_limit = params.get("time_limit", 10000)

        self.graph = G

        self.graph_order = len(G.nodes)

        self.model = gb.Model()
        self.model.setParam('TimeLimit', self.time_limit)

        self.solution = {}

    def solve(self):
        self._start_timer()

        vars_dict = {}

        for node in self.graph.nodes:
            vars_dict[node] = self.model.addVar(name=f'v_{node}', vtype=gb.GRB.BINARY)
        
        C_i = [vars_dict[i] + vars_dict[j] -2*vars_dict[i]*vars_dict[j] for i,j in self.graph.edges]

        self.model.setObjective(sum(C_i), gb.GRB.MAXIMIZE)

        self.model.optimize()

        reverse_map = {v:k for k, v in vars_dict.items()}

        return self.model, [int(vars_dict[n].x) for n in self.graph.nodes]

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        self.solution[
            "graph_probabilities"
        ] = self.model.theta_layer.weight.detach().tolist()

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i
