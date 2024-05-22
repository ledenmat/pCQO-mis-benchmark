import torch
from torch.func import grad, vmap

import networkx as nx
from networkx import Graph

import torch.optim as optim

import time
from lib.Solver import Solver

def loss_function(Matrix_X, adjacency_matrix_tensor,adjacency_matrix_tensor_comp, gamma, beta):

    summed_weights = Matrix_X.sum()

    second_term = (gamma/2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X)

    third_term = (beta/2) * ((Matrix_X.T @ (adjacency_matrix_tensor_comp)) @ Matrix_X)

    loss = -summed_weights + second_term - third_term

    return loss


def normalize_adjacency_matrix(graph):
    # Get the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    # Convert to PyTorch tensor
    adjacency_matrix = torch.Tensor(adjacency_matrix)

    # Calculate the degree matrix
    degree_matrix = torch.diag(torch.tensor(list(dict(graph.degree()).values())))

    # Normalize the adjacency matrix
    normalized_adjacency = (
        torch.inverse(torch.sqrt(degree_matrix))
        @ adjacency_matrix
        @ torch.inverse(torch.sqrt(degree_matrix)).to_dense()
    )

    return normalized_adjacency


class Quadratic_Batch(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()

        self.learning_rate = params.get("learning_rate", 0.001)

        self.number_of_steps = params.get("number_of_steps", 10000)

        self.graph = G

        self.beta = params.get("beta", 1)

        self.gamma = params.get("gamma", 625)

        self.batch_size = params.get("batch_size", 256)

        self.steps_per_batch = params.get("steps_per_batch", 350)

        self.log_every_n_steps = params.get("log_every_n_steps", self.steps_per_batch)

        self.graphs_per_optimizer = params.get("graphs_per_optimizer", 128)

        self.threshold = params.get("threshold", 0.0)

        self.seed = 113

        self.graph_order = len(G.nodes)

        self.solution = {}

        self.normalize = params.get("normalize", False)

        self.combine = params.get("combine", False)

        self.value_initializer = torch.nn.init.uniform_

    def solve(self):
        # Obtain A_G and A_G hat (and/or N_G and N_G hat)

        self._start_timer()

        if not self.normalize or self.combine:
            adjacency_matrix_dense = torch.Tensor(
                nx.adjacency_matrix(self.graph).todense()
            ).to_dense()
            adjacency_matrix_comp_dense = torch.Tensor(
                nx.adjacency_matrix(nx.complement(self.graph)).todense()
            ).to_dense()
        if self.normalize or self.combine:
            normalized_adjacency_matrix_dense = normalize_adjacency_matrix(self.graph)
            normalized_adjacency_matrix_comp_dense = normalize_adjacency_matrix(
                nx.complement(self.graph)
            )
        if self.combine:
            adjacency_matrix_dense = torch.stack(
                (adjacency_matrix_dense, normalized_adjacency_matrix_dense), dim=0
            )
            adjacency_matrix_comp_dense = torch.stack(
                (adjacency_matrix_comp_dense, normalized_adjacency_matrix_comp_dense),
                dim=0,
            )
        elif self.normalize:
            adjacency_matrix_dense = normalized_adjacency_matrix_dense
            adjacency_matrix_comp_dense = normalized_adjacency_matrix_comp_dense

        # Optimization loop:
        # Initialization:
        torch.manual_seed(self.seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device: ", device)

        Matrix_X = torch.empty((self.batch_size, self.graph_order))

        for batch in range(self.batch_size):
            Matrix_X.data[batch, :] = self.value_initializer(
                torch.empty((self.graph_order))
            )

        Matrix_X = Matrix_X.to(device)

        Matrix_X = Matrix_X.requires_grad_(True)

        gamma = torch.tensor(self.gamma, device=device)

        beta = torch.tensor(self.beta, device=device)

        learning_rate_alpha = self.learning_rate

        number_of_iterations_T = self.number_of_steps

        adjacency_matrix_tensor = adjacency_matrix_dense.to(device)
        adjacency_matrix_tensor_comp = adjacency_matrix_comp_dense.to(device)

        # Define Optimizer over matrix X
        with torch.no_grad():
            parts = torch.split(Matrix_X, self.graphs_per_optimizer)

        optimizers = []

        for part in parts:
            optimizers.append(optim.Adam([part], learning_rate_alpha))

        # optimizer = optim.Adam(parts, lr=learning_rate_alpha)
        # optimizer2 = optim.Adam([part_two], lr=learning_rate_alpha)

        best_MIS = 0
        MIS = []
        batched_IS = [[]] * self.batch_size
        test_runtime = False

        solution_path = []
        solution_times = []

        steps_to_best_MIS = 0

        per_sample_grad_funct = vmap(grad(loss_function), in_dims=(0, None, None, None, None))

        if device == "cuda:0":
            torch.cuda.synchronize()

        for iteration_t in range(number_of_iterations_T):

            if test_runtime:
                start_time = time.time()

            for optimizer in optimizers:
                optimizer.zero_grad()

            # optimizer.zero_grad()
            # optimizer2.zero_grad()

            if test_runtime:
                torch.cuda.synchronize()
                zero_grad_time = time.time()
                print("time to zero gradients:", zero_grad_time - start_time)

            per_sample_gradients = torch.split(per_sample_grad_funct(Matrix_X, adjacency_matrix_tensor, adjacency_matrix_tensor_comp, gamma, beta), self.graphs_per_optimizer)


            with torch.no_grad():
                for i, part in enumerate(parts):
                    part.grad = per_sample_gradients[i]


            if test_runtime:
                torch.cuda.synchronize()
                backpropagation_time = time.time()
                print(
                    "time to compute back propagation:",
                    backpropagation_time - zero_grad_time,
                )
            
            
            for optimizer in optimizers:
                optimizer.step()
            # optimizer.step()  # Update the parameters
            # optimizer2.step()

            if test_runtime:
                torch.cuda.synchronize()
                optim_step_time = time.time()
                print(
                    "time to compute step time:", optim_step_time - backpropagation_time
                )

            # Box-constraining:
            Matrix_X.data[Matrix_X >= 1] = 1
            Matrix_X.data[Matrix_X <= 0] = 0

            if test_runtime:
                torch.cuda.synchronize()
                box_constraint_time = time.time()
                print(
                    "time to perform box constraining:",
                    box_constraint_time - optim_step_time,
                )

            if (iteration_t + 1) % self.steps_per_batch == 0:
                
                masks = Matrix_X.data[:, :] > self.threshold

                for batch_id, mask in enumerate(masks):
                    indices = mask.nonzero(as_tuple=True)[0].tolist()
                    subgraph = self.graph.subgraph(indices)
                    local_IS = indices

                    # If no MIS, move one
                    # if MIS_checker(MIS, G)[0] is False: MIS = []
                    if any(subgraph.edges()):
                        local_IS = []

                    batched_IS[batch_id] = local_IS

                for batch_id, IS in enumerate(batched_IS):
                    IS_length = len(IS)
                    if IS_length > best_MIS:
                        steps_to_best_MIS = iteration_t
                        best_MIS = IS_length
                        MIS = IS

                self._stop_timer()
                solution_path.append(best_MIS)
                solution_times.append(self.solution_time)

                # # Restart X and the optimizer to search at a different point in [0,1]^n
                with torch.no_grad():
                    for batch in range(self.batch_size):
                        Matrix_X.data[batch, :] = self.value_initializer(
                            torch.empty((self.graph_order))
                        )
                        Matrix_X = Matrix_X.to(device).requires_grad_(True)
                        

            if (iteration_t + 1) % self.log_every_n_steps == 0:
                print(
                    f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, MIS Size: {best_MIS}"
                )
                

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        print(solution_path, solution_times)

        print(f"Steps to best MIS: {steps_to_best_MIS}")

        self.solution["graph_mask"] = MIS
        self.solution["size"] = best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T
