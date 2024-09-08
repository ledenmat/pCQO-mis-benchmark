import torch
from torch.func import grad, vmap
import torch.optim as optim
import networkx as nx
from networkx import Graph
import time
from lib.Solver import Solver


def three_term_loss_function(
    Matrix_X, adjacency_matrix_tensor, adjacency_matrix_tensor_comp, gamma, beta
):
    """
    Computes the loss function for the three-term CQO variant.

    Parameters:
        Matrix_X (torch.Tensor): The matrix of variable values.
        adjacency_matrix_tensor (torch.Tensor): The adjacency matrix of the original graph.
        adjacency_matrix_tensor_comp (torch.Tensor): The adjacency matrix of the complement graph.
        gamma (float): Regularization parameter for the adjacency matrix of the original graph.
        beta (float): Regularization parameter for the adjacency matrix of the complement graph.

    Returns:
        torch.Tensor: The computed loss value.
    """
    summed_weights = Matrix_X.sum()

    second_term = (gamma / 2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X)
    third_term = (beta / 2) * ((Matrix_X.T @ (adjacency_matrix_tensor_comp)) @ Matrix_X)

    loss = -summed_weights + second_term - third_term

    return loss


def two_term_loss_function(
    Matrix_X, adjacency_matrix_tensor, gamma
):
    """
    Computes the loss function for the two-term QO variant.

    Parameters:
        Matrix_X (torch.Tensor): The matrix of variable values.
        adjacency_matrix_tensor (torch.Tensor): The adjacency matrix of the original graph.
        gamma (float): Regularization parameter for the adjacency matrix of the original graph.

    Returns:
        torch.Tensor: The computed loss value.
    """
    summed_weights = Matrix_X.sum()
    second_term = (gamma / 2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X)

    loss = -summed_weights + second_term

    return loss


def normalize_adjacency_matrix(graph):
    """
    Normalizes the adjacency matrix of a graph.

    Parameters:
        graph (networkx.Graph): The graph whose adjacency matrix will be normalized.

    Returns:
        torch.Tensor: The normalized adjacency matrix as a PyTorch tensor.
    """
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


class pCQOMIS(Solver):
    """
    Solver for the Maximum Independent Set (MIS) problem using a Quadratic Optimization approach with 
    a three-term or two-term loss function.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            - number_of_steps (int, optional): Number of training steps. Defaults to 10000.
            - beta (float, optional): Loss function parameter. Defaults to 1.
            - number_of_terms (str, optional): Type of loss function to use ("two" or "three"). Defaults to "three".
            - gamma (float, optional): Loss function parameter. Defaults to 775.
            - batch_size (int, optional): Number of graphs per batch. Defaults to 256.
            - steps_per_batch (int, optional): Number of optimization steps per batch. Defaults to 350.
            - output_interval (int, optional): Interval for outputting progress. Defaults to steps_per_batch.
            - graphs_per_optimizer (int, optional): Number of graphs per optimizer. Defaults to 128.
            - threshold (float, optional): Threshold for binarization of solutions. Defaults to 0.0.
            - seed (int, optional): Random seed for initialization. Defaults to 113.
            - normalize (bool, optional): Whether to normalize adjacency matrices. Defaults to False.
            - combine (bool, optional): Whether to combine original and normalized adjacency matrices. Defaults to False.
            - value_initializer (str, optional): Method for initializing values ("random" or "degree"). Defaults to "random".
            - value_initializer_std (float, optional): Standard deviation for random initialization (only applies to "degree-based" initializations). Defaults to 2.25.
            - test_runtime (bool, optional): Whether to test runtime performance. Defaults to False.
            - save_sample_path (bool, optional): Whether to save the sample path. Defaults to False.
            - adam_beta_1 (float, optional): Beta1 parameter for Adam optimizer. Defaults to 0.9.
            - adam_beta_2 (float, optional): Beta2 parameter for Adam optimizer. Defaults to 0.999.
    """

    def __init__(self, G: Graph, params):
        """
        Initializes the pCQOMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver including learning_rate, number_of_steps, beta, etc.
        """
        super().__init__()

        self.learning_rate = params.get("learning_rate", 0.001)
        self.number_of_steps = params.get("number_of_steps", 10000)
        self.graph = G
        self.beta = params.get("beta", 1)
        self.number_of_terms = params.get("number_of_terms", "three")
        self.gamma = params.get("gamma", 775)
        self.batch_size = params.get("batch_size", 256)
        self.steps_per_batch = params.get("steps_per_batch", 350)
        self.output_interval = params.get("output_interval", self.steps_per_batch)
        self.graphs_per_optimizer = params.get("graphs_per_optimizer", 128)
        self.threshold = params.get("threshold", 0.0)
        self.seed = 113
        self.graph_order = len(G.nodes)
        self.solution = {}
        self.normalize = params.get("normalize", False)
        self.combine = params.get("combine", False)
        self.value_initializer = params.get("value_initializer", "random")
        self.value_initializer_std = params.get("value_initializer_std", 2.25)
        self.test_runtime = params.get("test_runtime", False)
        self.save_sample_path = params.get("save_sample_path", False)
        self.adam_beta_1 = params.get("adam_beta_1", 0.9)
        self.adam_beta_2 = params.get("adam_beta_2", 0.999)

        ### Value Initializer Code
        if self.value_initializer == "random":
            self.value_initializer = torch.nn.init.uniform_
        elif self.value_initializer == "degree":
            mean_vector = []
            degrees = dict(self.graph.degree())

            # Find the maximum degree
            max_degree = max(degrees.values())

            for _, degree in self.graph.degree():
                degree_init = 1 - degree / max_degree
                mean_vector.append(degree_init)

            min_degree_initialization = max(mean_vector)

            for i in range(len(mean_vector)):
                mean_vector[i] = mean_vector[i] / min_degree_initialization

            self.value_initializer = lambda _: torch.normal(
                mean=torch.Tensor(mean_vector), std=self.value_initializer_std
            )
        ### End Value Initializer Code

    def solve(self):
        """
        Solves the Maximum Independent Set (MIS) problem using a neural network approach.

        The method performs the following steps:
        1. Prepares adjacency matrices (original and complement) and normalizes them if needed.
        2. Initializes the variable matrix and sets up the optimizer.
        3. Performs optimization to train the model.
        4. Checks for independent sets in the current solution and updates the best MIS found.
        5. Records runtime statistics if required.
        6. Outputs the results including the best MIS and the number of steps taken.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (torch.Tensor): Tensor where 1s denote nodes in the MIS.
                - size (int): Size of the best MIS found.
                - number_of_steps (int): Total number of training steps performed.
                - steps_to_best_MIS (int): Number of steps to reach the best MIS.
        """
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
        if self.number_of_terms == "three":
            adjacency_matrix_tensor_comp = adjacency_matrix_comp_dense.to(device)

        # Define Optimizer over matrix X
        with torch.no_grad():
            parts = torch.split(Matrix_X, self.graphs_per_optimizer)

        optimizers = []

        for part in parts:
            optimizers.append(optim.Adam([part], learning_rate_alpha, betas=(self.adam_beta_1, self.adam_beta_2)))

        best_MIS = 0
        MIS = []

        zero_grad_time_cum = 0
        per_sample_grad_time_cum = 0
        optim_step_time_cum = 0
        box_constraint_time_cum = 0
        is_check_time_cum = 0
        restart_time_cum = 0
        initializations_solved = 0

        if self.save_sample_path:
            solution_path = []
            solution_times = []

        output_tensors = []
        steps_to_best_MIS = 0

        if self.number_of_terms == "three":
            per_sample_grad_funct = vmap(
                grad(three_term_loss_function), in_dims=(0, None, None, None, None)
            )
        else:
            per_sample_grad_funct = vmap(
                grad(two_term_loss_function), in_dims=(0, None, None)
            )

        if device == "cuda:0":
            torch.cuda.synchronize()

        for iteration_t in range(number_of_iterations_T):

            if self.test_runtime:
                start_time = time.time()

            for optimizer in optimizers:
                optimizer.zero_grad()

            if self.test_runtime:
                torch.cuda.synchronize()
                zero_grad_time = time.time()
                zero_grad_time_cum += zero_grad_time - start_time

            if self.number_of_terms == "three":
                per_sample_gradients = torch.split(
                    per_sample_grad_funct(
                        Matrix_X,
                        adjacency_matrix_tensor,
                        adjacency_matrix_tensor_comp,
                        gamma,
                        beta,
                    ),
                    self.graphs_per_optimizer,
                )
            else:
                per_sample_gradients = torch.split(
                    per_sample_grad_funct(
                        Matrix_X,
                        adjacency_matrix_tensor,
                        gamma,
                    ),
                    self.graphs_per_optimizer,
                )

            with torch.no_grad():
                for i, part in enumerate(parts):
                    part.grad = per_sample_gradients[i]

            if self.test_runtime:
                torch.cuda.synchronize()
                per_sample_gradient_time = time.time()
                per_sample_grad_time_cum += per_sample_gradient_time - zero_grad_time

            for optimizer in optimizers:
                optimizer.step()

            if self.test_runtime:
                torch.cuda.synchronize()
                optim_step_time = time.time()
                optim_step_time_cum += optim_step_time - per_sample_gradient_time

            # Box-constraining:
            Matrix_X.data[Matrix_X >= 1] = 1
            Matrix_X.data[Matrix_X <= 0] = 0

            if self.test_runtime:
                torch.cuda.synchronize()
                box_constraint_time = time.time()
                box_constraint_time_cum += box_constraint_time - optim_step_time

            if (iteration_t + 1) % self.steps_per_batch == 0:
                masks = Matrix_X.data[:,:].bool().float().clone()
                output_tensors.append(masks)
                n = self.graph_order

                masks = masks.to(device)
                indices_to_replace = []

                for batch_id, X_torch_binarized in enumerate(masks):
                    if X_torch_binarized.sum() != 0 and (X_torch_binarized.T @ adjacency_matrix_tensor @ X_torch_binarized) == 0:
                        # we have an IS. Next, we check if this IS is maximal based on the proof of the second theorem: Basically, we are checking if it is a local min based on the fixed point definition:
                        # if for some gradient update, we are still at the boundary, then we have maximal IS
                        X_torch_binarized_update = X_torch_binarized - 0.1*(-torch.ones(n, device=device) + (n*adjacency_matrix_tensor - adjacency_matrix_tensor_comp)@X_torch_binarized)
                        # Projection to [0,1]
                        X_torch_binarized_update[X_torch_binarized_update>=1] =1
                        X_torch_binarized_update[X_torch_binarized_update<=0] =0
                        if torch.equal(X_torch_binarized, X_torch_binarized_update):
                            initializations_solved += 1
                            indices_to_replace.append(batch_id)
                            # we have a maximal IS:
                            MIS = torch.nonzero(X_torch_binarized).squeeze()
                            # Exit the function with True
                            if len(MIS) > best_MIS:
                                steps_to_best_MIS = iteration_t + 1
                                best_MIS = len(MIS)
                                MIS = MIS
                
                if self.test_runtime:
                    torch.cuda.synchronize()
                    IS_check_time = time.time()
                    is_check_time_cum += IS_check_time - box_constraint_time

                if self.save_sample_path:
                    self._stop_timer()
                    solution_path.append(best_MIS)
                    solution_times.append(self.solution_time)

                # Restart X and the optimizer to search at a different point in [0,1]^n
                with torch.no_grad():
                    for batch in range(self.batch_size):
                        Matrix_X.data[batch, :] = self.value_initializer(
                            torch.empty((self.graph_order))
                        )
                        Matrix_X = Matrix_X.to(device).requires_grad_(True)

                if self.test_runtime:
                    torch.cuda.synchronize()
                    restart_time = time.time()
                    restart_time_cum += restart_time - IS_check_time

            if (iteration_t + 1) % self.output_interval == 0:
                print(
                    f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, MIS Size: {best_MIS}"
                )

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        if self.save_sample_path:
            print(solution_path, solution_times)

        print(f"Steps to best MIS: {steps_to_best_MIS}")

        if self.test_runtime:
            print(f"Total time spent zero-ing gradients: {zero_grad_time_cum}")
            print(f"Total time spent computing per-sample gradients: {per_sample_grad_time_cum}")
            print(f"Total time spent taking optimizer steps: {optim_step_time_cum}")
            print(f"Total time spent box constraining input: {box_constraint_time_cum}")
            print(f"Total time spent checking for IS and updating: {is_check_time_cum}")
            print(f"Total time spent restarting initializations: {restart_time_cum}")

        print(f"Initializations solved: {initializations_solved}")

        self.solution["graph_mask"] = MIS
        self.solution["size"] = best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T
        self.solution["steps_to_best_MIS"] = steps_to_best_MIS
