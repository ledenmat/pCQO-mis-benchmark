import torch
from torch import Tensor

from itertools import combinations

from lib.Solver import Solver
from models.datalessnet import DatalessNet

import networkx as nx
from networkx import Graph

# PARAMS:

## max_steps: How many steps would you like to use to train your model?
## selection_criteria: At what threshold should theta values be selected?
## learning_rate: At what learning rate should your optimizer start at?
# OUTPUTS:

## solution:
### graph_mask: numpy array filled with 0s and 1s. 1s signify nodes in the MIS. Use this as a mask on the original graph to see MIS solution
### graph_probabilities: numpy array with theta weight results for the entire graph
### size: size of MIS
### number_of_steps: number of steps needed to find the solution


class CustomActivation(torch.nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(1000000 * x)


class DNNMIS(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()
        self.selection_criteria = params.get("selection_criteria", 0.45)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.max_steps = params.get("number_of_steps", 10000)
        self.use_cpu = params.get("use_cpu", False)
        self.solve_interval = params.get("solve_interval", 100)
        self.weight_decay = params.get("weight_decay", 0)

        self.graph = G

        self.temperature_schedule = torch.tensor(
            params.get("temp_schedule", ([0] * 55 + [0.5] * 55) * self.max_steps)
        )

        self.graph_order = len(G.nodes)

        self.model = DatalessNet(G)

        self.model.gamma = (self.graph_order * (self.graph_order - 1)) / (
            2 * self.model.graph_size
        )
        
        self.model.gamma = 1

        self.model.update_gamma()

        self.model.activation = CustomActivation()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.loss_fn = lambda predicted: predicted

        self.x = torch.ones(self.graph_order)

        self.solution = {}

    def solve(self):
        self._start_timer()

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu"
        )
        print("using device: ", device)

        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.temperature_schedule = self.temperature_schedule.to(device)

        MIS_mask = []
        MIS_size = 0


        for i in range(self.max_steps):
            self.optimizer.zero_grad()

            self.model.temperature = self.temperature_schedule[i]

            predicted: Tensor = self.model(self.x)

            output = self.loss_fn(predicted)

            output.backward()
            self.optimizer.step()

            with torch.no_grad():
                graph_probs = self.model.theta_layer.weight.data.tolist()

            graph_mask = [0 if x < self.selection_criteria else 1 for x in graph_probs]

            indices = [i for i, x in enumerate(graph_mask) if x == 1]

            if i % self.solve_interval == 0 and len(indices) > MIS_size + 1:
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
                if IS_size > MIS_size:
                    MIS_size = IS_size
                    MIS_mask = graph_mask
                    print(f"Found MIS of size: {MIS_size}")

            if i % 5000 == 0:
                print(
                    f"Training step: {i}, MIS size: {MIS_size}, Output: {predicted.item():.4f}"
                )

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        self.solution[
            "graph_probabilities"
        ] = self.model.theta_layer.weight.detach().tolist()

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i
