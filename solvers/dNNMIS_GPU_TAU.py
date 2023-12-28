import torch
from torch import Tensor

from itertools import combinations

from lib.Solver import Solver
from models.datalessnet import DatalessNet

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
        self.selection_criteria = params.get("selection_criteria", 0.8)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.max_steps = params.get("max_steps", 10000)
        self.convergence_std = params.get("convergence_std", 0.000000001)
        self.use_cpu = params.get("use_cpu", False)
        
        self.graph = G

        self.temperature_schedule = torch.tensor(
            (
                [0] * 55
                + [0.1] * 55
                + [0.2] * 55
                + [0.3] * 55
                + [0.4] * 55
                + [0.5] * 55
            )
            * 10000
        )


        self.graph_order = len(G.nodes)

        self.model = DatalessNet(G)

        self.model.gamma = (self.graph_order * (self.graph_order -1 )) / (2 * self.model.graph_size)
        self.model.gamma = 2
        self.model.update_gamma()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.loss_fn = lambda predicted, desired: predicted - desired

        self.loss_fn = lambda predicted, _: predicted

        self.x = torch.ones(self.graph_order)
        self.objective = torch.tensor(-(self.graph_order**2) / 2)
        self.solution = {}

    def solve(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu"
        )
        print("using device: ", device)

        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.objective = self.objective.to(device)
        self.temperature_schedule = self.temperature_schedule.to(device)

        self._start_timer()

        MIS_mask = []
        MIS_size = 0

        for i in range(self.max_steps):
            self.optimizer.zero_grad()

            self.model.temperature = self.temperature_schedule[i]

            predicted: Tensor = self.model(self.x)

            output = self.loss_fn(predicted, self.objective)

            output.backward()
            self.optimizer.step()

            if i % 50 == 0:
                with torch.no_grad():
                    graph_probs = self.model.theta_layer.weight.data.tolist()

                print(graph_probs)

                graph_mask = [
                    0 if x < self.selection_criteria else 1 for x in graph_probs
                ]

                indices = [i for i, x in enumerate(graph_mask) if x == 1]

                IS_size = sum(graph_mask) if MIS_checker(indices, self.graph) is True else 0

                if IS_size > MIS_size:
                    MIS_size = IS_size
                    MIS_mask = graph_mask

                print(
                    f"Training step: {i}, MIS size: {MIS_size}, Output: {predicted.item():.4f}, Desired Output: {self.objective.item():.4f}"
                )

        self._stop_timer()

        self.solution[
            "graph_probabilities"
        ] = self.model.theta_layer.weight.detach().tolist()

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i


def MIS_checker(MIS_list, G):
    pairs = list(combinations(MIS_list, 2))
    IS_CHECKER = True
    if len(MIS_list) > 1:
        for pair in pairs:
            if (pair[0], pair[1]) in G.edges or (pair[1], pair[0]) in G.edges:
                IS_CHECKER = False
                break
    return IS_CHECKER
