import torch
from copy import deepcopy
import numpy
from lib.Solver import Solver
from lib.datalessNN import datalessNN_graph_params, datalessNN_module

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


class DNNMIS_SUBGRAPH_V2(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        self.selection_set = params.get("selection_set", [0.2, 0.8])
        self.learning_rate = params.get("learning_rate", 0.001)
        self.convergence_epsilon = params.get("convergence_epsilon", 0.00001)
        self.subgraph_solve_interval = params.get("subgraph_solve_interval", 20000)
        self.max_epochs = params.get("max_epochs", 100000)
        self.solution = {"subgraphs": [], "subgraph_probabilities": []}

        # Generate default graph parameters
        nn_params = datalessNN_graph_params(self.G)

        # Initialize the base neural network
        self.NN = datalessNN_module(
            nn_params["theta_tensor"],
            nn_params["layer_2_weights"],
            nn_params["layer_2_biases"],
            nn_params["layer_3_weights"],
        )

        # Compute k
        self.graph_order = len(self.G.nodes)

    def solve(self):
        optimizer = torch.optim.Adam(self.NN.parameters(), lr=self.learning_rate)

        x = torch.ones(self.graph_order)
        y_desired = torch.Tensor([-(self.graph_order**2) / 2])

        self._start_timer()

        fpi = 0
        five_previous = torch.zeros(5)

        epoch_count = 0

        while epoch_count < self.max_epochs:
            epoch_count += 1
            y_predicted = self.NN(x)

            loss = y_predicted - y_desired

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch_count % 500 == 0:
                print(
                    f"Epoch: {epoch_count}, Output: {y_predicted.item():.4f}, Desired Output: {y_desired.item():.4f}"
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

            if epoch_count % self.subgraph_solve_interval == 0:
                node_selection_small = self.NN[0].weight > self.selection_set[0]
                node_selection_large = self.NN[0].weight < self.selection_set[1]

                node_selection_mask = node_selection_small & node_selection_large
                node_selection_list = (
                    torch.nonzero(node_selection_mask).squeeze().tolist()
                )
                torch.set_printoptions(sci_mode=False)
                print(self.NN[0].weight)
                subgraph = self.G.subgraph(
                    node_selection_list
                )
                self.solution["subgraphs"].append(deepcopy(subgraph))
                if len(subgraph.nodes) > 1 and len(subgraph.edges) > 0:
                    subgraph_instance = DNNMIS_SUBGRAPH_V2(
                        subgraph,
                        {
                            "learning_rate": self.learning_rate,
                            "selection_set": self.selection_set,
                            "max_epochs": self.max_epochs - epoch_count,
                        },
                    )
                    subgraph_instance.solve()

                    subgraph_solution_probs = torch.Tensor(subgraph_instance.solution[
                        "graph_probabilities"
                    ])

                    for idn, node in enumerate(node_selection_list):
                        with torch.no_grad():
                            self.NN[0].weight.data[node] = subgraph_solution_probs[idn]
                else:
                    for idn, node in enumerate(node_selection_list):
                        if node_selection_list[idn]:
                            with torch.no_grad():
                                self.NN[0].weight.data[node] = 0
        self._stop_timer()

        self.solution["graph_probabilities"] = self.NN[0].weight.detach().numpy()

        graph_mask = (
            self.solution["graph_probabilities"] > self.selection_set[1]
        ) * self.solution["graph_probabilities"]
        graph_mask[graph_mask < 1] = 0

        self.solution["graph_mask"] = graph_mask
        self.solution["size"] = numpy.count_nonzero(self.solution["graph_mask"] >= 1)
        self.solution["number_of_steps"] = epoch_count

    @staticmethod
    def convergence_check(current, last_five, error):
        return all(torch.abs(current - last_five) < error)
