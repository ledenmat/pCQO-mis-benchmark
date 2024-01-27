import torch

import networkx as nx

import torch.optim as optim

import time
from lib.Solver import Solver

import networkx as nx
from networkx import Graph

def loss_function(adjacency_matrix_tensor,adjacency_matrix_tensor_comp, Matrix_X, gamma, beta):
    ## without edges of the comp graph:
    # loss = -Matrix_X.sum() + (gamma/2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X)

    ## with edges of the comp graph:
    loss = -Matrix_X.sum() + (gamma/2) * (Matrix_X.T @ (adjacency_matrix_tensor) @ Matrix_X) - (beta/2) * (Matrix_X.T @ (adjacency_matrix_tensor_comp) @ Matrix_X)
    return loss

def normalize_adjacency_matrix(adjacency_graph, degree_graph):
    # Get the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(adjacency_graph).todense()

    # Convert to PyTorch tensor
    adjacency_matrix = torch.Tensor(adjacency_matrix)

    # Calculate the degree matrix
    degree_matrix = torch.diag(torch.tensor(list(dict(degree_graph.degree()).values())))

    # Normalize the adjacency matrix
    normalized_adjacency = torch.inverse(torch.sqrt(degree_matrix)) @ adjacency_matrix @ torch.inverse(torch.sqrt(degree_matrix))
    
    return normalized_adjacency

class Quadratic(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()
        self.learning_rate = params.get("learning_rate", 0.001)
        self.max_steps = params.get("max_steps", 10000)

        self.graph = G

        self.beta = 0

        self.gamma = 0

        self.graph_order = len(G.nodes)

        self.solution = {}

        self.normalize = False

        self.rongrong = False

    def solve(self):
        adjacency_matrix = normalize_adjacency_matrix(self.graph, self.graph)
        adjacency_matrix_dense = adjacency_matrix.to_dense()

        if not self.normalize:
            adjacency_matrix = nx.adjacency_matrix(self.graph)
            adjacency_matrix_dense = adjacency_matrix.todense()
        adjacency_matrix_tensor = torch.tensor(adjacency_matrix_dense, dtype=torch.float32)

        ### Obtain the A_G_hat matrix

        adjacency_matrix_comp = normalize_adjacency_matrix(nx.complement(self.graph), nx.complement(self.graph))
        adjacency_matrix_dense_comp = adjacency_matrix_comp.to_dense()
        if not self.normalize:
            adjacency_matrix_comp = nx.adjacency_matrix(nx.complement(self.graph))
            adjacency_matrix_dense_comp = adjacency_matrix_comp.todense()
        if self.rongrong:
            adjacency_matrix_comp = normalize_adjacency_matrix(nx.complement(self.graph), self.graph)
            adjacency_matrix_dense_comp = adjacency_matrix_comp.to_dense()
        adjacency_matrix_tensor_comp = torch.tensor(adjacency_matrix_dense_comp, dtype=torch.float32)

        # Optimization loop:
        # Initialization:
        torch.manual_seed(115)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("using device: ", device)

        Matrix_X = torch.ones((self.graph_order), requires_grad=True, device=device)
        # Matrix_X = torch.rand((n), requires_grad=True, device=device)
        X_ini  = Matrix_X.data.clone()

        # dict for saving... I am thinking of diversification by looking at previous initializations... Still under investigation
        dict_of_inits = {}

        # This is obtained to get a sense of how far are we from the init
        #gamma = torch.tensor(50.0, requires_grad=True)

        learning_rate_alpha = self.learning_rate
        number_of_iterations_T = self.max_steps

        adjacency_matrix_tensor = adjacency_matrix_tensor.to(device)
        adjacency_matrix_tensor_comp = adjacency_matrix_tensor_comp.to(device)

        # Define Optimizer over matrix X
        optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        Best_MIS = 0
        MIS = []
        test_runtime = False

        improve_iteration = 0

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._start_timer()
        self._stop_timer()

        for iteration_t in range(number_of_iterations_T):
            
            if test_runtime:
                start_time = time.time()

            loss = loss_function(adjacency_matrix_tensor,adjacency_matrix_tensor_comp, Matrix_X, gamma = self.gamma, beta = self.beta)

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
                print("time to compute back propagation:", backpropagation_time - zero_grad_time)

            optimizer.step()  # Update the parameters

            if test_runtime:
                torch.cuda.synchronize()
                optim_step_time = time.time()
                print("time to compute step time:", optim_step_time - backpropagation_time)

            # Box-constraining:
            Matrix_X.data[Matrix_X>=1] =1
            Matrix_X.data[Matrix_X<=0] =0

            b = Matrix_X.data > 0.05
            indices = b.nonzero(as_tuple=True)[0].tolist()

            subgraph = self.graph.subgraph(indices)

            if test_runtime:
                torch.cuda.synchronize()
                box_constraint_time = time.time()
                print("time to perform box constraining:", box_constraint_time - optim_step_time)


            # Report the current MIS
            # MIS = []
            # for node in G.nodes:
            #     if Matrix_X[node] >0.0:
            #       MIS.append(node)
            IS = indices

            
            if test_runtime:
                torch.cuda.synchronize()
                MIS_generation_time = time.time()
                print("time to perform MIS selection:", MIS_generation_time - box_constraint_time)

            # If no MIS, move one
            # if MIS_checker(MIS, G)[0] is False: MIS = []
            if any(subgraph.edges()): IS = []

            if test_runtime:
                torch.cuda.synchronize()
                MIS_check_time = time.time()
                print("time to perform MIS checking:", MIS_check_time - MIS_generation_time)

            # Iteration logger every XX iterations:
            if (iteration_t + 1) % 1000 == 0:
                print(f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, MIS Size: {Best_MIS}")
                #print(f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, Loss: {loss.item()}, grad norm: {torch.norm(Matrix_X.grad).item()}")
                #print(f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, Loss: {loss.item()}")
            if len(IS) > 0:
                if improve_iteration == 0:
                    lr_scheduler.step()
                improve_iteration += 1
    
                if len(IS) > Best_MIS:
                    Best_MIS = len(IS)
                    MIS = IS
                    self._stop_timer()

                # print("+++++++++++ A MIS is found with size: ", [[[len(IS)]]], "++++ BEST so far",[Best_MIS], "Dist from Init: =",l2_norm.item())

                # # Restart X and the optimizer to search at a different point in [0,1]^n

                if (improve_iteration > 50):
                    improve_iteration = 0

                    Matrix_X = torch.nn.init.uniform_(torch.empty((self.graph_order), requires_grad=True, device=device))
                    optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)
                    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

        self.solution["graph_mask"] = MIS
        self.solution["size"] = Best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T