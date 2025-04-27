import torch
import os
from torch.func import vmap
import networkx as nx
import subprocess
from networkx import Graph
import time
from lib.Solver import Solver
from pathlib import Path
import logging
import numpy

module_directory = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

class pCQOMIS_MGD(Solver):
    def __init__(self, G: Graph, params: dict):
        """
        Solver for the Maximum Independent Set (MIS) problem using a Quadratic Optimization approach with 
        a three-term or two-term loss function.

        Parameters:
            G (networkx.Graph): The graph on which the MIS problem will be solved.
            params (dict): Dictionary containing solver parameters:
                - learning_rate (float, required): Learning rate for the optimizer.
                - momentum (float, required): Momentum for the optimizer.
                - number_of_total_steps (int, required): Number of training steps across all batches.
                - number_of_steps_per_batch (int, required): Number of training steps per batch.
                - gamma (float, required): Loss function parameter.
                - gamma_prime (float, required): Loss function parameter.
                - batch_size (int, required): Number of graphs per batch.
                - sampling_stddev (float, required): Standard deviation for sampling.
                - output_interval (int, required): Interval for outputting results. Outputs every number of batches.
                - initialization_vector (str, optional): Binary vector representing a non-maximized solution for the graph. If not provided, initialization vector is degree based.
                - pcqomis_path (str, optional): Path to the pCQO-MIS executable. Defaults to "../external/pcqo_mis".
        """
        super().__init__()
        self.G = G
        self.learning_rate = params.get('learning_rate')
        self.momentum = params.get('momentum')
        self.number_of_total_steps = params.get('number_of_total_steps')
        self.number_of_steps_per_batch = params.get('number_of_steps_per_batch')
        self.gamma = params.get('gamma')
        self.gamma_prime = params.get('gamma_prime')
        self.batch_size = params.get('batch_size')
        self.sampling_stddev = params.get('sampling_stddev')
        self.output_interval = params.get('output_interval')
        self.initialization_vector = params.get('initialization_vector')
        self.pcqomis_path = params.get('pcqomis_path', os.path.join(module_directory, "../external/pcqo_mis"))
        self.solution = {}

        if self.learning_rate is None or self.momentum is None or self.number_of_total_steps is None or \
            self.number_of_steps_per_batch is None or self.gamma is None or self.gamma_prime is None or \
            self.batch_size is None or self.sampling_stddev is None or self.output_interval is None:
            raise ValueError("All required parameters must be provided in the params dictionary.")

    def solve(self):

        temp_graph_path = f"./temp_pcqo_mis_dimacs_{time.time()}"
        temp_graph_os_path = Path(temp_graph_path)
        temp_result_path = f"./temp_pcqo_mis_solution_{time.time()}"
        temp_result_os_path = Path(temp_result_path)

        # Convert networkx graph to DIMACS format
        self.networkx_to_dimacs(self.G, temp_graph_path)

        # start time
        self._start_timer()

        # Build the command for the solver
        pcqomis_command = [
            self.pcqomis_path,
            temp_graph_path,
            str(self.learning_rate),
            str(self.momentum),
            str(self.number_of_total_steps),
            str(self.number_of_steps_per_batch),
            str(self.gamma),
            str(self.gamma_prime),
            str(self.batch_size),
            str(self.sampling_stddev),
            str(self.output_interval),
        ]

        if self.initialization_vector is not None:
            pcqomis_command.append(self.initialization_vector)

        with open(temp_result_path, "w") as result_file:
            subprocess.run(pcqomis_command, shell=False, stdout=result_file, text=True)

        # extract solution
        with open(temp_result_path, "r") as result_file:
            lines = result_file.readlines()
            if len(lines) >= 3:
                self.solution["size"] = int(lines[-3].strip())
                self.solution_time = float(lines[-2].strip())
                self.solution["graph_mask"] = numpy.array(lines[-1].strip().split(), dtype=int)
            else:
                raise ValueError("Result file does not contain enough lines to extract solution size and time.")
            
        temp_graph_os_path.unlink(missing_ok=True)
        temp_result_os_path.unlink(missing_ok=True)

    @staticmethod
    def networkx_to_dimacs(G, output_file):
        with open(output_file, "w") as f:
            # Write the header
            f.write("p EDGE {} {}\n".format(G.number_of_nodes(), G.number_of_edges()))
            # Write all edges
            for u, v in G.edges():
                f.write("e {} {}\n".format(u + 1, v + 1))

if __name__ == "__main__":
        # Create a simple example graph
    G = networkx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    # Set parameters for the ReduMIS solver
    params = {"time_limit": 10}  # 10 seconds time limit

    # Initialize and solve the MIS problem
    solver = pCQOMIS_MGD(G, params)
    solver.solve()