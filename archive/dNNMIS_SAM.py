import torch
import numpy
from lib.Solver import Solver
from archive.sam import SAM
from lib.datalessNN import datalessNN_graph_params, datalessNN_module
from lib.layer_constraints import ZeroOneClamp

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


class DNNMIS_SAM(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        self.selection_criteria = params.get("selection_criteria", 0.8)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.max_steps = params.get("max_steps", 10000)
        self.convergence_epsilon = params.get("convergence_epsilon", 0.00001)
        self.solution = {}

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

        even_spacing = self.max_steps // 3
        training_stages = [
            {"steps": even_spacing, "rho": 0.1},
            {"steps": even_spacing, "rho": 0.01},
            {"steps": even_spacing, "rho": 0.001},
        ]

        for stage in training_stages:
            ################# optimization
            # good and fast results on most is found at initial_learning_rate = 0.001
            base_optimizer = (
                torch.optim.Adam
            )  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(
                NN.parameters(), base_optimizer, lr=self.learning_rate, rho=stage["rho"]
            )

            for i in range(stage["steps"]):
                y_predicted = NN(x)

                loss = y_predicted - y_desired

                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)

                NN[0].apply(theta_constraint)

                if i % 500 == 0:
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

                loss.backward()

                optimizer.second_step(zero_grad=True)

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
