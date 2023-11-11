import torch
from copy import deepcopy
import numpy
from lib.Solver import Solver
from solvers.dNNMIS import DNNMIS
from lib.datalessNN import datalessNN_graph_params, datalessNN_module
from lib.layer_constraints import ZeroOneClamp

# PARAMS:

## max_steps: How many steps would you like to use to train your model?
## selection_criteria: At what threshold should theta values be selected?
## learning_rate: At what learning rate should your optimizer start at?
## convergence_epsilon: To what value epsilon do you want to check convergence to?
## number_of_stages: How many subgraphs should we attempt to solve within max_steps?
## max_subgraph_steps: How many steps should be used to solve each subgraph?

# OUTPUTS:

## solution:
### graph_mask: numpy array filled with 0s and 1s. 1s signify nodes in the MIS. Use this as a mask on the original graph to see MIS solution
### graph_probabilities: numpy array with theta weight results for the entire graph
### size: size of MIS
### number_of_steps: number of steps needed to find the solution
### subgraphs: array of subgraphs in order of early to late stages
### subgraph_probabilities: array of theta weight results for the subgraphs


class DNNMIS_SUBGRAPH(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        self.selection_criteria = params.get("selection_criteria", 0.8)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.convergence_epsilon = params.get("convergence_epsilon", 0.00001)
        self.number_of_stages = params.get("number_of_stages", 3)
        self.max_subgraph_steps = params.get("max_subgraph_steps", 1000)
        self.max_steps = params.get("max_steps", 10000)
        self.solution = {"subgraphs": [], "subgraph_probabilities": []}

    def solve(self):
        nn_params = datalessNN_graph_params(self.G)
        NN = datalessNN_module(
            nn_params["theta_tensor"],
            nn_params["layer_2_weights"],
            nn_params["layer_2_biases"],
            nn_params["layer_3_weights"],
        )

        graph_order = len(self.G.nodes)

        optimizer = torch.optim.Adam(NN.parameters(), lr=self.learning_rate)

        theta_constraint = ZeroOneClamp()

        x = numpy.ones(graph_order)
        y_desired = torch.Tensor([-(graph_order**2) / 2])

        self._start_timer()

        fpi = 0
        five_previous = numpy.zeros(5)

        even_spacing = self.max_steps // self.number_of_stages
        stages = [even_spacing] * self.number_of_stages

        for stage in stages:
            for i in range(stage):
                y_predicted = NN(x)

                loss = y_predicted - y_desired

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                NN[0].apply(theta_constraint)

                print(
                    f"Training step: {i}, Output: {y_predicted.item():.4f}, Desired Output: {y_desired.item():.4f}"
                )

                output_prediction = y_predicted.item()
                if fpi >= 5:
                    if self.convergence_check(
                        output_prediction, five_previous, self.convergence_epsilon
                    ):
                        break
                    fpi = 0
                five_previous[fpi] = output_prediction
                fpi += 1

            x_non_star = NN[0].weight.detach().numpy()
            node_selection_small = numpy.array(
                [
                    i
                    for i in range(len(x_non_star))
                    if x_non_star[i] < self.selection_criteria
                ]
            )
            node_selection_large = numpy.array(
                [i for i in range(len(x_non_star)) if x_non_star[i] > 0.2]
            )
            node_selection = numpy.intersect1d(
                node_selection_large, node_selection_small
            )
            subgraph = self.G.subgraph(node_selection)
            self.solution["subgraphs"].append(deepcopy(subgraph))
            subgraph_solution = subgraph
            if len(subgraph.nodes) > 1 and len(subgraph.edges) > 0:
                subgraph_instance = DNNMIS(
                    subgraph,
                    {
                        "learning_rate": self.learning_rate,
                        "selection_criteria": self.selection_criteria,
                        "max_steps": self.max_subgraph_steps,
                    },
                )
                subgraph_instance.solve()

                subgraph_solution_probs = subgraph_instance.solution[
                    "graph_probabilities"
                ]

                for idn, node in enumerate(node_selection):
                    x_non_star[node] = subgraph_solution_probs[idn]
                with torch.no_grad():
                    NN[0].weight.data = torch.Tensor(x_non_star)

            self.solution["subgraph_probabilities"].append(deepcopy(subgraph_solution))

        self._stop_timer()

        self.solution["graph_probabilities"] = NN[0].weight.detach().numpy()

        graph_mask = (
            self.solution["graph_probabilities"] > self.selection_criteria
        ) * self.solution["graph_probabilities"]
        graph_mask[graph_mask < 1] = 0

        self.solution["graph_mask"] = graph_mask
        self.solution["size"] = numpy.count_nonzero(self.solution["graph_mask"] == 1)
        self.solution["number_of_steps"] = i

    @staticmethod
    def convergence_check(current, last_five, error):
        return numpy.all(abs(current - last_five) < error)
