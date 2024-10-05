import torch
from torch.func import vmap
import networkx as nx
from networkx import Graph
import time
from lib.Solver import Solver
import logging

logger = logging.getLogger(__name__)


def three_term_grad_function(
    Matrix_X, adjacency_matrix_tensor, adjacency_matrix_tensor_comp, gamma, gamma_prime
):
    """
    Computes the gradient for the three-term CQO variant.

    Parameters:
        Matrix_X (torch.Tensor): The matrix of variable values.
        adjacency_matrix_tensor (torch.Tensor): The adjacency matrix of the original graph.
        adjacency_matrix_tensor_comp (torch.Tensor): The adjacency matrix of the complement graph.
        gamma (float): Regularization parameter for the adjacency matrix of the original graph.
        gamma_prime (float): Regularization parameter for the adjacency matrix of the complement graph.

    Returns:
        torch.Tensor: The computed gradient value.
    """
    grad = -1 + (gamma) * (adjacency_matrix_tensor @ Matrix_X) - (gamma_prime) * (adjacency_matrix_tensor_comp @ Matrix_X)

    return grad


def two_term_grad_function(
    Matrix_X, adjacency_matrix_tensor, gamma
):
    """
    Computes the gradient for the two-term QO variant.

    Parameters:
        Matrix_X (torch.Tensor): The matrix of variable values.
        adjacency_matrix_tensor (torch.Tensor): The adjacency matrix of the original graph.
        gamma (float): Regularization parameter for the adjacency matrix of the original graph.

    Returns:
        torch.Tensor: The computed gradient value.
    """
    grad = -1 + (gamma) * (adjacency_matrix_tensor @ Matrix_X)

    return grad

def velocity_update_function(
        vector_x, gradient, velocity, momentum, learning_rate
):
    new_velocity = momentum * velocity + learning_rate * gradient

    vector_x = vector_x - new_velocity

    return vector_x, new_velocity


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


class pCQOMIS_GD(Solver):
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
            - threshold (float, optional): Threshold for binarization of solutions. Defaults to 0.0.
            - seed (int, optional): Random seed for initialization. Defaults to 113.
            - normalize (bool, optional): Whether to normalize adjacency matrices. Defaults to False.
            - combine (bool, optional): Whether to combine original and normalized adjacency matrices. Defaults to False.
            - value_initializer (str, optional): Method for initializing values ("random" or "degree"). Defaults to "random".
            - value_initializer_std (float, optional): Standard deviation for random initialization (only applies to "degree-based" initializations). Defaults to 2.25.
            - test_runtime (bool, optional): Whether to test runtime performance. Defaults to False.
            - save_sample_path (bool, optional): Whether to save the sample path. Defaults to False.
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
        self.momentum = params.get("momentum", 0.9)

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

            self.initial_value_initializer = lambda _: torch.normal(
                mean=torch.Tensor(mean_vector), std=self.value_initializer_std
            )
            self.value_initializer = lambda mean, test: torch.normal(
                out=test, mean=mean, std=self.value_initializer_std
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
        logger.info("using device: %s", device)


        Matrix_X = torch.empty((self.batch_size, self.graph_order), device=device, requires_grad=False)
        velocity_matrix = torch.zeros((self.batch_size, self.graph_order), device=device, requires_grad=False)

        for batch in range(self.batch_size):
            Matrix_X.data[batch, :] = self.initial_value_initializer(
                torch.empty((self.graph_order))
            )

        gamma = torch.tensor(self.gamma, device=device)
        beta = torch.tensor(self.beta, device=device)
        learning_rate = torch.tensor(self.learning_rate, device=device)
        momentum = torch.tensor(self.momentum, device=device)
        number_of_iterations_T = self.number_of_steps

        adjacency_matrix_tensor = adjacency_matrix_dense.to(device)
        adjacency_matrix_tensor_comp = adjacency_matrix_comp_dense.to(device)

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
                three_term_grad_function, in_dims=(0, None, None, None, None)
            )
        else:
            per_sample_grad_funct = vmap(
                two_term_grad_function, in_dims=(0, None, None)
            )

        per_sample_velocity_update_funct = vmap(
                velocity_update_function, in_dims=(0, 0, 0, None, None)
            )

        if device == "cuda:0":
            torch.cuda.synchronize()

        for iteration_t in range(number_of_iterations_T):

            if self.test_runtime:
                start_time = time.time()

            if self.test_runtime:
                torch.cuda.synchronize()
                zero_grad_time = time.time()
                zero_grad_time_cum += zero_grad_time - start_time

            if self.number_of_terms == "three":
                per_sample_gradients = per_sample_grad_funct(
                        Matrix_X,
                        adjacency_matrix_tensor,
                        adjacency_matrix_tensor_comp,
                        gamma,
                        beta,
                    )
            else:
                per_sample_gradients = per_sample_grad_funct(
                        Matrix_X,
                        adjacency_matrix_tensor,
                        gamma,
                    )

            if self.test_runtime:
                torch.cuda.synchronize()
                per_sample_gradient_time = time.time()
                per_sample_grad_time_cum += per_sample_gradient_time - zero_grad_time

            Matrix_X, velocity_matrix = per_sample_velocity_update_funct(
                Matrix_X,
                per_sample_gradients,
                velocity_matrix,
                momentum,
                learning_rate
            )

            if self.test_runtime:
                torch.cuda.synchronize()
                optim_step_time = time.time()
                optim_step_time_cum += optim_step_time - per_sample_gradient_time

            # Box-constraining:
            Matrix_X = Matrix_X.clamp(min=0, max=1)

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
                                track_this = X_torch_binarized
                
                if self.test_runtime:
                    torch.cuda.synchronize()
                    IS_check_time = time.time()
                    is_check_time_cum += IS_check_time - box_constraint_time

                if self.save_sample_path:
                    self._stop_timer()
                    solution_path.append(best_MIS)
                    solution_times.append(self.solution_time)

                # Restart X and the optimizer to search at a different point in [0,1]^n
                for batch in range(self.batch_size):
                    Matrix_X.data[batch, :] = self.value_initializer(track_this,
                        torch.empty((self.graph_order)).to(device)
                    )
                    # Matrix_X.data[batch, :] = self.initial_value_initializer(
                    #     torch.empty((self.graph_order)).to(device)
                    # )

                if self.test_runtime:
                    torch.cuda.synchronize()
                    restart_time = time.time()
                    restart_time_cum += restart_time - IS_check_time

            if (iteration_t + 1) % self.output_interval == 0:
                logger.info("Step %d/%d, IS: %s, lr: %s, MIS Size: %s", iteration_t + 1, number_of_iterations_T, MIS, learning_rate, best_MIS)


        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        logger.info("Steps to best MIS: %s", steps_to_best_MIS)

        if self.test_runtime:
            logger.info("Total time spent zero-ing gradients: %s", zero_grad_time_cum)
            logger.info("Total time spent computing per-sample gradients: %s", per_sample_grad_time_cum)
            logger.info("Total time spent taking optimizer steps: %s", optim_step_time_cum)
            logger.info("Total time spent box constraining input: %s", box_constraint_time_cum)
            logger.info("Total time spent checking for IS and updating: %s", is_check_time_cum)
            logger.info("Total time spent restarting initializations: %s", restart_time_cum)


        logger.info("Initializations solved: %s", initializations_solved)

        self.solution["graph_mask"] = MIS
        self.solution["size"] = best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T
        self.solution["steps_to_best_MIS"] = steps_to_best_MIS
