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

def batched_loss_function(adjacency_matrix_tensor, adjacency_matrix_tensor_comp, Matrix_X_batch, gamma, beta):

    # Matrix_X_batch is now assumed to be of shape [batch_size, num_nodes, 1]

    # Compute the loss for the batch

    term1 = -Matrix_X_batch.sum(dim=1)  # Sum over nodes, keep batch dimension

    print(Matrix_X_batch[0,:,0])

    term2 = (gamma / 2) * torch.bmm(torch.bmm(Matrix_X_batch.transpose(1, 2), adjacency_matrix_tensor), Matrix_X_batch).squeeze(1) 

    term3 = (beta / 2) * torch.bmm(torch.bmm(Matrix_X_batch.transpose(1, 2), adjacency_matrix_tensor_comp), Matrix_X_batch).squeeze(1) 

    # Calculate total loss for each element in the batch and then mean

    loss = term1 + term2 - term3

    return loss


def normalize_adjacency_matrix(graph):
    # Get the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    # Convert to PyTorch tensor
    adjacency_matrix = torch.Tensor(adjacency_matrix)

    # Calculate the degree matrix
    degree_matrix = torch.diag(torch.tensor(list(dict(graph.degree()).values())))

    # Normalize the adjacency matrix
    normalized_adjacency = torch.inverse(torch.sqrt(degree_matrix)) @ adjacency_matrix @ torch.inverse(torch.sqrt(degree_matrix)).to_dense()
    
    return normalized_adjacency

class Quadratic(Solver):
    def __init__(self, G: Graph, params):
        super().__init__()
        self.learning_rate = params.get("learning_rate", 0.001)
        self.number_of_steps = params.get("number_of_steps", 10000)

        self.graph = G

        self.beta = 1

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

        self.value_initializer = params.get("value_initializer", torch.nn.init.uniform_)

    def solve(self):
        # Obtain A_G and A_G hat (and/or N_G and N_G hat)

        self._start_timer()

        if not self.normalize or self.combine:
            adjacency_matrix_dense = torch.Tensor(nx.adjacency_matrix(self.graph).todense()).to_dense()
            adjacency_matrix_comp_dense = torch.Tensor(nx.adjacency_matrix(nx.complement(self.graph)).todense()).to_dense()
        if self.normalize or self.combine:
            normalized_adjacency_matrix_dense = normalize_adjacency_matrix(self.graph)
            normalized_adjacency_matrix_comp_dense = normalize_adjacency_matrix(nx.complement(self.graph))
        if self.combine:
            adjacency_matrix_dense = torch.stack((adjacency_matrix_dense, normalized_adjacency_matrix_dense), dim=0)
            adjacency_matrix_comp_dense = torch.stack((adjacency_matrix_comp_dense, normalized_adjacency_matrix_comp_dense), dim=0)
        elif self.normalize:
            adjacency_matrix_dense = normalized_adjacency_matrix_dense
            adjacency_matrix_comp_dense = normalized_adjacency_matrix_comp_dense

        adj_matrix_batch_size = self.batch_size // 2 if self.combine else self.batch_size
        
        adjacency_matrix_dense = adjacency_matrix_dense.repeat(adj_matrix_batch_size, 1, 1)
        adjacency_matrix_comp_dense = adjacency_matrix_comp_dense.repeat(adj_matrix_batch_size, 1, 1)
            

        # Optimization loop:
        # Initialization:
        torch.manual_seed(self.seed)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("using device: ", device)

        Matrix_X = torch.ones((self.graph_order,1))

        Matrix_X = Matrix_X.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        Matrix_X = Matrix_X.to(device)

        Matrix_X = Matrix_X.requires_grad_(True)

        gamma = torch.tensor(self.gamma, device=device)

        beta = torch.tensor(self.beta, device=device)

        learning_rate_alpha = self.learning_rate

        number_of_iterations_T = self.number_of_steps

        adjacency_matrix_tensor = adjacency_matrix_dense.to(device)
        adjacency_matrix_tensor_comp = adjacency_matrix_comp_dense.to(device)

        # Define Optimizer over matrix X
        optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_gamma)

        best_MIS = 0
        MIS = []
        batched_IS = [[]] * self.batch_size
        test_runtime = False

        steps_to_best_MIS = 0

        explore_iteration = 0
        improve_iteration = 0

        if device == "cuda:0":
            torch.cuda.synchronize()

        for iteration_t in range(number_of_iterations_T):
            
            if test_runtime:
                start_time = time.time()

            loss = batched_loss_function(adjacency_matrix_tensor,adjacency_matrix_tensor_comp, Matrix_X, gamma = gamma, beta = beta)

            if test_runtime:
                torch.cuda.synchronize()
                loss_time = time.time()
                print("time to compute loss function:", loss_time - start_time)

            optimizer.zero_grad()  # Clear gradients for the next iteration

            if test_runtime:
                torch.cuda.synchronize()
                zero_grad_time = time.time()
                print("time to zero gradients:", zero_grad_time - loss_time)

            loss.sum().backward()  # Backpropagation

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

            if test_runtime:
                torch.cuda.synchronize()
                box_constraint_time = time.time()
                print("time to perform box constraining:", box_constraint_time - optim_step_time)



            # Iteration logger every XX iterations:
            if (iteration_t + 1) % 1000 == 0:
                print(f"Step {iteration_t + 1}/{number_of_iterations_T}, IS: {MIS}, lr: {learning_rate_alpha}, MIS Size: {best_MIS}")
            
            if explore_iteration == self.explore_split:
                lr_scheduler.step()
                improve_iteration = 0
            
            if improve_iteration == self.improve_split:
                explore_iteration = 0

                masks = Matrix_X.data[:,:,0] > self.threshold

                for batch_id, mask in enumerate(masks):
                    indices = mask.nonzero(as_tuple=True)[0].tolist()
                    subgraph = self.graph.subgraph(indices)
                    local_IS = indices

                    # If no MIS, move one
                    # if MIS_checker(MIS, G)[0] is False: MIS = []
                    if any(subgraph.edges()): local_IS = []

                    batched_IS[batch_id] = local_IS
                
                if test_runtime:
                    torch.cuda.synchronize()
                    MIS_generation_time = time.time()
                    print("time to perform MIS selection:", MIS_generation_time - box_constraint_time)

                replace_batch = []

                for batch_id, IS in enumerate(batched_IS):
                    IS_length = len(IS)
                    if IS_length > best_MIS:
                        steps_to_best_MIS = iteration_t
                        best_MIS = IS_length
                        MIS = IS
                    
                    if IS_length > 0:
                        replace_batch.append(batch_id)


                if test_runtime:
                    torch.cuda.synchronize()
                    MIS_check_time = time.time()
                    print("time to perform MIS checking:", MIS_check_time - MIS_generation_time)

                # # Restart X and the optimizer to search at a different point in [0,1]^n
                with torch.no_grad():
                    for batch in replace_batch:
                        Matrix_X.data[batch, :, :] = self.value_initializer(torch.empty((self.graph_order, 1)))
                optimizer = optim.Adam([Matrix_X], lr=learning_rate_alpha)
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_gamma)
            
            improve_iteration += 1
            explore_iteration += 1

        if device == "cuda:0":
            torch.cuda.synchronize()
        self._stop_timer()

        print(f"Steps to best MIS: {steps_to_best_MIS}")

        self.solution["graph_mask"] = MIS
        self.solution["size"] = best_MIS
        self.solution["number_of_steps"] = number_of_iterations_T