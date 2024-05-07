import os
import pickle

import networkx as nx
from networkx import Graph

class SDP():
    def __init__(self, initialization_set_file, params):
        super().__init__()

        with open(initialization_set_file, "rb") as f:
            self.intialization_set  = pickle.load(f)

        self.gamma = params.get("gamma", 625)

        self.batch_size = params.get("batch_size", 150)

        self.threshold = params.get("threshold", 0.0)

        self.seed = 113

        self.lr_gamma = params.get("lr_gamma", 0.2)

        self.graph_order = len(G.nodes)

        self.solution = {}

        self.normalize = params.get("normalize", False)

        self.combine = params.get("combine", False)

        self.explore_split = params.get("train_split", 85)

        self.improve_split = params.get("improve_split", 25)

        self.value_initializer = params.get("value_initializer", torch.nn.init.uniform_)

    def solve(self):
        