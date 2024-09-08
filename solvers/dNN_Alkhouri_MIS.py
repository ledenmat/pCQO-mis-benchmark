import torch
from torch import Tensor

from lib.Solver import Solver
from models.datalessnet import DatalessNet

import networkx as nx

# PARAMS:

## max_steps: How many steps would you like to use to train your model?
## selection_criteria: At what threshold should theta values be selected?
## learning_rate: At what learning rate should your optimizer start at?
## convergence_epsilon: To what value epsilon do you want to check convergence to?

# OUTPUTS:

## solution:
### graph_mask: numpy array filled with 0s and 1s. 1s signify nodes in the MIS. Use this as a mask on the original graph to see MIS solution
### graph_probabilities: numpy array with theta weight results for the entire graph
### size: size of MIS
### number_of_steps: number of steps needed to find the solution


class DNNMIS(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.selection_criteria = params.get("selection_criteria", 0.5)
        self.learning_rate = params.get("learning_rate", 0.0001)
        self.max_steps = params.get("max_steps", 100000)
        self.use_cpu = params.get("use_cpu", False)

        self.graph = G

        self.graph_order = len(G.nodes)
        print(self.graph_order)

        self.model = DatalessNet(G)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = lambda predicted, desired : predicted - desired
        
        self.x = torch.ones(self.graph_order)
        self.objective = torch.tensor(-(self.graph_order**2) / 2)
        self.solution = {}

    def solve(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        print("using device: ", device)

        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.objective = self.objective.to(device)

        self._start_timer()

        for i in range(self.max_steps):
            self.optimizer.zero_grad()

            predicted: Tensor = self.model(self.x)

            output = self.loss_fn(predicted, self.objective)

            output.backward()
            self.optimizer.step()

            if i % 500 == 0:
                print(
                    f"Training step: {i}, Output: {predicted.item():.4f}, Desired Output: {self.objective.item():.4f}"
                )

        self._stop_timer()

        self.solution["graph_probabilities"] = self.model.theta_layer.weight.detach().tolist()

        graph_mask = [0 if x < self.selection_criteria else 1 for x in self.solution["graph_probabilities"]]
        indices = [i for i, x in enumerate(graph_mask) if x == 1]

        subgraph = self.graph.subgraph(indices)
        subgraph = nx.Graph(subgraph)
        while len(subgraph) > 0:
            degrees = dict(subgraph.degree())
            max_degree_nodes = [
                node
                for node, degree in degrees.items()
                if degree == max(degrees.values())
            ]

            if (
                len(max_degree_nodes) == 0
                or subgraph.degree(max_degree_nodes[0]) == 0
            ):
                break  # No more nodes to remove or all remaining nodes have degree 0

            subgraph.remove_node(max_degree_nodes[0])
        IS_size = len(subgraph)
        MIS_size = IS_size
        MIS_mask = graph_mask
        print(f"Found MIS of size: {MIS_size}")

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i
        self.solution["steps_to_best_MIS"] = 0