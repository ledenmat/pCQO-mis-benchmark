import torch

import networkx as nx

import torch.optim as optim

import time
from lib.Solver import Solver

import networkx as nx
from networkx import Graph

import numpy as np

from mosek.fusion import Model, Domain, ObjectiveSense, Expr

from itertools import combinations


def batched_loss_function(
    adjacency_matrix_tensor,
    adjacency_matrix_tensor_comp,
    Matrix_X,
    s_SDP_v_tensor,
    W_SDP_uv_tensor,
    gamma,
    beta,
):

    loss = (
        -(Matrix_X.T @ s_SDP_v_tensor)
        + (gamma / 2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X)
        - (beta / 2)
        * (Matrix_X.T @ (adjacency_matrix_tensor_comp * W_SDP_uv_tensor) @ Matrix_X)
    )

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


def solve_SDP(graph):
    n = len(graph)
    matrix_of_ones = np.ones((n, n))

    with Model("SDP_init") as m:
        X = m.variable("X", [n, n], Domain.inPSDCone())
        objective_expr = Expr.dot(matrix_of_ones, X)
        m.objective("obj", ObjectiveSense.Maximize, objective_expr)
        m.constraint("tr", Expr.sum(X.diag()), Domain.equalsTo(1.0))
        for edge in graph.edges:
            i, j = edge
            m.constraint(f"con_{i}_{j}", X.index(i, j), Domain.equalsTo(0.0))
            m.constraint(f"con_{j}_{i}", X.index(j, i), Domain.equalsTo(0.0))

        m.solve()
        X_solution = X.level()
        X_solution = np.array(X_solution)
        X_solution = X_solution.reshape(n, n)

    X_SDP_solution = X_solution

    ## Coding W_SDP_uv:
    W_SDP_uv = torch.zeros(n, n)

    # create a dictionary of key = edge (u,v) in G' and value is equation to X(u,v)
    G_hat_edges_strength_dict = {}

    for v in range(n):
        for u in range(n):
            if (v, u) not in graph.edges() and u != v:
                G_hat_edges_strength_dict[(v, u)] = X_SDP_solution[v, u]

    # Normalize
    values_array = np.array(list(G_hat_edges_strength_dict.values()))
    normalized_values_MINMAX = (values_array - np.min(values_array)) / (
        np.max(values_array) - np.min(values_array)
    )
    G_hat_edges_strength_dict_normalized_MINMAX = {
        key: normalized_values_MINMAX[i]
        for i, key in enumerate(G_hat_edges_strength_dict)
    }  ######## USE THIS

    # Assign to W_SDP_uv
    for v in range(n):
        for u in range(n):
            if (v, u) not in graph.edges() and v != u:
                W_SDP_uv[v, u] = G_hat_edges_strength_dict_normalized_MINMAX[(v, u)]

    ## Coding s_SDP_v and X_SDP_init:

    sol_diag = np.diag(X_SDP_solution)
    s_SDP_v = sol_diag / np.max(sol_diag)
    X_SDP_init = (sol_diag - np.min(sol_diag)) / (np.max(sol_diag) - np.min(sol_diag))

    ### Converting to torch tensors: Convert s_SDP_v and W_SDP_uv to torch tensors
    s_SDP_v_tensor = torch.tensor(s_SDP_v, dtype=torch.float32)
    X_SDP_init_tensor = torch.tensor(X_SDP_init, dtype=torch.float32)
    W_SDP_uv_tensor = torch.tensor(W_SDP_uv, dtype=torch.float32)

    return [s_SDP_v_tensor, X_SDP_init_tensor, W_SDP_uv_tensor]


def MIS_checker(MIS_list, G):
    pairs = list(combinations(MIS_list, 2))
    IS_CHECKER = True
    if len(MIS_list) > 1:
        for pair in pairs:
            if (pair[0], pair[1]) in G.edges or (pair[1], pair[0]) in G.edges:
                IS_CHECKER = False
                break
    # is it a MIS or IS?
    # if MIS_list is an IS:
    if IS_CHECKER:
        # print("Its an IS")
        # obtain a list of all niebouring nodes
        list_of_all_Ns_in_MIS_list = []
        for node in MIS_list:
            list_of_all_Ns_in_MIS_list.append(list(G.neighbors(node)))

        flattened_list_of_all_Ns_in_MIS_list = [
            item for sublist in list_of_all_Ns_in_MIS_list for item in sublist
        ]
        # print(flattened_list_of_all_Ns_in_MIS_list)

        list_of_nodes_inG_but_not_in_MIStoCHeck = [
            item for item in list(G.nodes) if item not in MIS_list
        ]

        MIS_CHECKER = True
        for node in list_of_nodes_inG_but_not_in_MIStoCHeck:
            if node not in flattened_list_of_all_Ns_in_MIS_list:
                MIS_CHECKER = False
                break
    else:
        IS_CHECKER = False
        MIS_CHECKER = False
    return IS_CHECKER, MIS_CHECKER


class Quadratic_SDP(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()
        self.learning_rate = params.get("learning_rate", 0.001)
        self.number_of_steps = params.get("number_of_steps", 10000)

        self.graph = G

        self.beta = params.get("beta", 1)

        self.gamma = params.get("gamma", 625)

        self.batch_size = params.get("batch_size", 150)

        self.threshold = params.get("threshold", 0.0)

        self.seed = 113

        self.lr_gamma = params.get("lr_gamma", 0.2)

        self.graph_order = len(G.nodes)

        self.solution = {}

        self.normalize = params.get("normalize", False)

        self.combine = params.get("combine", False)

        self.explore_split = params.get("train_split", 85)

        self.improve_split = params.get("improve_split", 25)

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

        [s_SDP_v_tensor, X_SDP_init_tensor, W_SDP_uv_tensor] = solve_SDP(self.graph)

        mean_vector = X_SDP_init_tensor
        covariance_matrix = 1 * torch.eye(len(mean_vector))

        # Optimization loop:
        # Initialization:
        torch.manual_seed(self.seed)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        print("using device: ", device)

        Matrix_X = X_SDP_init_tensor

        Matrix_X = Matrix_X.to(device)

        Matrix_X = Matrix_X.requires_grad_(True)

        s_SDP_v_tensor = s_SDP_v_tensor.to(device)

        W_SDP_uv_tensor = W_SDP_uv_tensor.to(device)

        gamma = torch.tensor(self.gamma, device=device)

        beta = torch.tensor(self.beta, device=device)

        learning_rate_alpha = self.learning_rate

        number_of_iterations_T = self.number_of_steps

        adjacency_matrix_tensor = adjacency_matrix_dense.to(device)
        adjacency_matrix_tensor_comp = adjacency_matrix_comp_dense.to(device)

        # Define Optimizer over matrix X
        optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)

        best_MIS = 0
        MIS = []
        batched_IS = [[]] * self.batch_size
        test_runtime = False

        steps_to_best_MIS = 0

        MIS_found = False

        if device == "cuda:0":
            torch.cuda.synchronize()

        for iteration_t in range(number_of_iterations_T):

            if test_runtime:
                start_time = time.time()

            loss = batched_loss_function(
                adjacency_matrix_tensor,
                adjacency_matrix_tensor_comp,
                Matrix_X,
                s_SDP_v_tensor,
                W_SDP_uv_tensor,
                gamma=gamma,
                beta=beta,
            )

            if test_runtime:
                torch.cuda.synchronize()
                loss_time = time.time()
                print("time to compute loss function:", loss_time - start_time)

            optimizer.zero_grad()  # Clear gradients for the next iteration

            if test_runtime:
                torch.cuda.synchronize()
                zero_grad_time = time.time()
                print("time to zero gradients:", zero_grad_time - loss_time)

            loss.backward()  # Backpropagation

            if test_runtime:
                torch.cuda.synchronize()
                backpropagation_time = time.time()
                print(
                    "time to compute back propagation:",
                    backpropagation_time - zero_grad_time,
                )

            optimizer.step()  # Update the parameters

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

            # Iteration logger every XX iterations:
            if (iteration_t + 1) % 1000 == 0:
                print(
                    f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, MIS Size: {best_MIS}"
                )

            # mask = Matrix_X.data > self.threshold

            # indices = mask.nonzero(as_tuple=True)[0].tolist()
            # subgraph = self.graph.subgraph(indices)
            # local_IS = indices

            # # If no MIS, move one
            # if any(subgraph.edges()):
            #     local_IS = []
                
            MIS = []
            for node in self.graph.nodes:
                if Matrix_X[node] >0.1:
                    MIS.append(node)

            # If no MIS, move one
            if MIS_checker(MIS, self.graph)[0] is False: MIS = []

            if test_runtime:
                torch.cuda.synchronize()
                MIS_generation_time = time.time()
                print(
                    "time to perform MIS selection:",
                    MIS_generation_time - box_constraint_time,
                )

            if MIS_checker(MIS, self.graph)[1] is True:
                if len(MIS) > best_MIS: # This is to save our current best so far
                    steps_to_best_MIS = iteration_t
                    best_MIS = len(MIS)

                print(f"MIS FOUND WITH SIZE {len(MIS)}, BEST SO FAR IS {best_MIS}")

            # if IS_length > best_MIS:
            #     steps_to_best_MIS = iteration_t
            #     best_MIS = IS_length
            #     MIS = local_IS
            #     MIS_found = True

                if test_runtime:
                    torch.cuda.synchronize()
                    MIS_check_time = time.time()
                    print(
                        "time to perform MIS checking:",
                        MIS_check_time - MIS_generation_time,
                    )
                with torch.no_grad():
                    Matrix_X.data = torch.normal(
                        mean=mean_vector, std=torch.sqrt(torch.diag(covariance_matrix))*.5
                    )
                    Matrix_X = Matrix_X.to(device).requires_grad_(True)
                    Matrix_X.data[Matrix_X >= 1] = 1
                    Matrix_X.data[Matrix_X <= 0] = 0
                Matrix_X = Matrix_X.to(device).requires_grad_(True)
                optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        print(f"Steps to best MIS: {steps_to_best_MIS}")

        self.solution["graph_mask"] = MIS
        self.solution["size"] = best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T
