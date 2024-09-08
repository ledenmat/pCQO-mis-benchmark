import torch
from torch import Tensor
from lib.Solver import Solver
from models.datalessnet import DatalessNet
import networkx as nx

class DNNMIS(Solver):
    """
    A solver class for finding the Maximum Independent Set (MIS) of a graph using a
    dataless neural network model. The neural network is trained to predict theta values
    which are then used to determine the MIS.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - max_steps (int, optional): Maximum number of training steps for the model. Defaults to 100000.
            - selection_criteria (float, optional): Threshold for selecting nodes based on theta values. Defaults to 0.5.
            - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            - use_cpu (bool, optional): Flag to use CPU for computations instead of GPU. Defaults to False.
    """

    def __init__(self, G, params):
        """
        Initializes the DNNMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver including max_steps, selection_criteria, learning_rate, and use_cpu.
        """
        super().__init__()
        self.selection_criteria = params.get("selection_criteria", 0.5)
        self.learning_rate = params.get("learning_rate", 0.0001)
        self.max_steps = params.get("max_steps", 100000)
        self.use_cpu = params.get("use_cpu", False)

        self.graph = G
        self.graph_order = len(G.nodes)
        print(self.graph_order)

        self.model = DatalessNet(G)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = lambda predicted, desired: predicted - desired
        
        self.x = torch.ones(self.graph_order)
        self.objective = torch.tensor(-(self.graph_order ** 2) / 2)
        self.solution = {}

    def solve(self):
        """
        Trains the neural network model to find the Maximum Independent Set (MIS) of the graph.

        The method performs the following steps:
        1. Trains the model for a specified number of steps.
        2. Evaluates the model to get theta values.
        3. Applies a selection criterion to determine which nodes are in the MIS.
        4. Constructs the MIS by iteratively removing nodes with the highest degree from the subgraph.
        5. Records the solution details including the graph mask, size of the MIS, and number of training steps.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (list of int): List where 1s denote nodes in the MIS.
                - graph_probabilities (list of float): Theta weight results for each node in the graph.
                - size (int): Size of the MIS.
                - number_of_steps (int): Number of training steps performed.
                - steps_to_best_MIS (int): Number of steps to reach the best MIS (currently set to 0).
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        print("using device: ", device)

        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.objective = self.objective.to(device)

        self._start_timer()

        for i in range(self.max_steps):
            self.optimizer.zero_grad()

            predicted: Tensor = self.model(self.x)

            output = self.loss_fn(predicted, self.objective)

            output.backward()
            self.optimizer.step()

            if i % 500 == 0:
                print(
                    f"Training step: {i}, Output: {predicted.item():.4f}, Desired Output: {self.objective.item():.4f}"
                )

        self._stop_timer()

        self.solution["graph_probabilities"] = self.model.theta_layer.weight.detach().tolist()

        graph_mask = [0 if x < self.selection_criteria else 1 for x in self.solution["graph_probabilities"]]
        indices = [i for i, x in enumerate(graph_mask) if x == 1]

        subgraph = self.graph.subgraph(indices)
        subgraph = nx.Graph(subgraph)
        while len(subgraph) > 0:
            degrees = dict(subgraph.degree())
            max_degree_nodes = [
                node
                for node, degree in degrees.items()
                if degree == max(degrees.values())
            ]

            if (
                len(max_degree_nodes) == 0
                or subgraph.degree(max_degree_nodes[0]) == 0
            ):
                break  # No more nodes to remove or all remaining nodes have degree 0

            subgraph.remove_node(max_degree_nodes[0])
        IS_size = len(subgraph)
        MIS_size = IS_size
        MIS_mask = graph_mask
        print(f"Found MIS of size: {MIS_size}")

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i
        self.solution["steps_to_best_MIS"] = 0
