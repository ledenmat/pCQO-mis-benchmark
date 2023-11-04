import torch
import numpy
from Solver import Solver
from sam import SAM
from datalessNN import datalessNN_graph_params, datalessNN_module
from layer_constraints import ZeroOneClamp

# PARAMS:

## max_steps: How many steps would you like to use to train your model?
## selection_criteria: At what threshold should theta values be selected?
## optimizer: Define optimizer parameters
### method: Options include: "Adam", "SGD"
### learning_rate: At what learning rate should your optimizer start at?

class DNNMIS_SAM(Solver):
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
            self.nn_params["layer_3_weights"])

    def solve(self):
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        theta_constraint = ZeroOneClamp()
        x = numpy.ones(len(self.nn_params["theta_tensor"]))
        y_desired = torch.Tensor([-self.G_order**2/2])
        
        self._start_timer()

        fpi = 0
        five_previous = numpy.zeros(5)

        even_spacing = self.max_steps // 3
        training_stages = [{"steps": even_spacing, "rho": 0.05},
                          {"steps": even_spacing, "rho": 0.25},
                          {"steps": even_spacing, "rho": 0.5}]

        for stage in training_stages:
            ################# optimization
            # good and fast results on most is found at initial_learning_rate = 0.001
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(self.nn.parameters(), base_optimizer, lr=self.learning_rate, rho=stage["rho"])

            for i in range(stage["steps"]):
                y_predicted = self.nn(x)

                loss = (y_predicted - y_desired)

                loss.backward(retain_graph = True)
                optimizer.first_step(zero_grad=True)

                self.nn[0].apply(theta_constraint)

                print(f"Training step: {i}, Output: {y_predicted.item():.4f}, Desired Output: {y_desired.item():.4f}")

                output_prediction = y_predicted.item()
                if fpi >= 5:
                    if self.convergence_check(output_prediction, five_previous, 0.00001):
                        break
                    fpi = 0
                five_previous[fpi] = output_prediction
                fpi += 1

                loss.backward()

                optimizer.second_step(zero_grad=True)

        self._stop_timer()

        self.solution = self.nn[0].weight.detach().numpy()
        self.number_of_steps = i
        self.solution_size = (self.solution > self.selection_criteria).sum()

    @staticmethod
    def convergence_check(current, last_five, error):
        return numpy.all(abs(current - last_five) < error)