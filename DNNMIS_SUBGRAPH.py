import torch
import numpy
from Solver import Solver
from DNNMIS import DNNMIS
from datalessNN import datalessNN_graph_params, datalessNN_module
from layer_constraints import ZeroOneClamp

# PARAMS:

## max_steps: How many steps would you like to use to train your model?
## selection_criteria: At what threshold should theta values be selected?
## optimizer: Define optimizer parameters
### method: Options include: "Adam", "SGD"
### learning_rate: At what learning rate should your optimizer start at?


class DNNMIS_SUBGRAPH(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        self.G_order = len(G.nodes)
        self.params = params
        self.selection_criteria = params["selection_criteria"]
        self.learning_rate = params["learning_rate"]
        self.max_steps = params["max_steps"]
        self.nn_params = datalessNN_graph_params(G)
        self.nn = datalessNN_module(
            self.nn_params["theta_tensor"],
            self.nn_params["layer_2_weights"],
            self.nn_params["layer_2_biases"],
            self.nn_params["layer_3_weights"],
        )

    def solve(self):
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        theta_constraint = ZeroOneClamp()
        x = numpy.ones(len(self.nn_params["theta_tensor"]))
        y_desired = torch.Tensor([-self.G_order**2 / 2])

        self._start_timer()

        fpi = 0
        five_previous = numpy.zeros(5)

        even_spacing = self.max_steps // 3
        stages = [even_spacing, even_spacing, even_spacing]

        for stage in stages:
            for i in range(stage):
                y_predicted = self.nn(x)

                loss = y_predicted - y_desired

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.nn[0].apply(theta_constraint)

                print(
                    f"Training step: {i}, Output: {y_predicted.item():.4f}, Desired Output: {y_desired.item():.4f}"
                )

                output_prediction = y_predicted.item()
                if fpi >= 5:
                    if self.convergence_check(
                        output_prediction, five_previous, 0.00001
                    ):
                        break
                    fpi = 0
                five_previous[fpi] = output_prediction
                fpi += 1

            x_non_star = self.nn[0].weight.detach().numpy()
            node_selection_small = numpy.array([i for i in range(len(x_non_star)) if x_non_star[i] < 0.8])
            node_selection_large = numpy.array([i for i in range(len(x_non_star)) if x_non_star[i] > 0.2])
            node_selection = numpy.intersect1d(node_selection_large, node_selection_small)
            subgraph = self.G.subgraph(node_selection)
            print(subgraph.nodes, subgraph.edges)
            if len(subgraph.nodes) > 1 and len(subgraph.edges) > 0:
                subgraph_instance = DNNMIS(
                    subgraph,
                    {"learning_rate": 0.001, "selection_criteria": 0.8, "max_steps": 5000},
                )
                print(subgraph)
                subgraph_instance.solve()
                for idn, node in enumerate(node_selection):
                    x_non_star[node] = subgraph_instance.solution[idn]
                print(x_non_star)
                with torch.no_grad():
                    self.nn[0].weight.data = torch.Tensor(x_non_star)

        self._stop_timer()

        self.solution = self.nn[0].weight.detach().numpy()
        self.number_of_steps = i
        self.solution_size = (self.solution > self.selection_criteria).sum()

    @staticmethod
    def convergence_check(current, last_five, error):
        return numpy.all(abs(current - last_five) < error)
