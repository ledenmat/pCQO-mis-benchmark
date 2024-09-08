import os
import subprocess
import re
from lib.Solver import Solver
import time
from pathlib import Path
import numpy
import networkx

module_directory = os.path.dirname(os.path.abspath(__file__))

class ReduMIS(Solver):
    """
    A solver class for finding the Maximum Independent Set (MIS) of a graph using the ReduMIS algorithm.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - seed (int, optional): Seed for randomization. Defaults to None.
            - time_limit (int, optional): Time limit (in seconds) for the algorithm to run. Defaults to None.
            - redumis_path (str, optional): Path to the ReduMIS executable. Defaults to "../external/redumis".
    """
    def __init__(self, G, params):
        """
        Initializes the ReduMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver, including optional settings for seed, time_limit, and redumis_path.
        """
        super().__init__()
        self.G = G
        self.seed = params.get("seed", None)
        self.time_limit = params.get("time_limit", None)
        self.redumis_path = params.get(
            "redumis_path", os.path.join(module_directory, "../external/redumis")
        )
        self.solution = {}

    def solve(self):
        """
        Executes the ReduMIS algorithm to find the Maximum Independent Set (MIS) of the graph.

        The method performs the following steps:
        1. Converts the input graph to METIS format and saves it to a temporary file.
        2. Constructs and executes the ReduMIS command with the specified parameters.
        3. Extracts the solution time from the ReduMIS log file.
        4. Reads and processes the solution from the result file.
        5. Cleans up temporary files used during execution.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (numpy.array): Array of 0s and 1s where 1s denote nodes in the MIS.
                - size (int): Size of the MIS.
        """
        # Define temporary file paths
        temp_graph_path = f"./temp_kamis_metis_{time.time()}"
        temp_graph_os_path = Path(temp_graph_path)
        temp_result_path = f"./temp_kamis_solution_{time.time()}"
        temp_result_os_path = Path(temp_result_path)

        # Convert networkx graph to METIS format
        self.networkx_to_metis(self.G, temp_graph_path)

        # Start timing
        self._start_timer()

        # Build ReduMIS command
        redumis_command = [
            self.redumis_path,
            temp_graph_path,
            f"--output={temp_result_path}",
        ]

        if self.time_limit is not None:
            redumis_command.append(f"--time_limit={self.time_limit}")

        if self.seed is not None:
            redumis_command.append(f"--seed={self.seed}")

        # Execute ReduMIS command
        with open("redumis.log", "w") as output_file:
            subprocess.run(redumis_command, shell=False, stdout=output_file, text=True)

        # Extract solution time from the log file
        solution_time_regex = r"(?<=Time found:).+([0-9]+.[0-9]+)"
        with open("redumis.log", "r") as f:
            for line in f:
                match = re.search(solution_time_regex, line)
                if match:
                    self.solution_time = float(match.group())

        # Read and process the result
        with open(temp_result_path, "r") as f:
            result = f.read().split("\n")
            result.pop()  # Remove the last empty line
            self.solution["graph_mask"] = numpy.array(result, dtype=int)
            self.solution["size"] = numpy.count_nonzero(self.solution["graph_mask"] == 1)

        # Clean up temporary files
        temp_graph_os_path.unlink(missing_ok=True)
        temp_result_os_path.unlink(missing_ok=True)

    @staticmethod
    def networkx_to_metis(G, output_file):
        """
        Converts a networkx graph to METIS format and writes it to a file.

        Args:
            G (networkx.Graph): The graph to be converted.
            output_file (str): Path to the file where the METIS formatted graph will be saved.

        The METIS format consists of:
        - The number of nodes and edges on the first line.
        - Each subsequent line represents the adjacency list of each node.
        """
        graph_order = len(G.nodes)
        graph_size = len(G.edges)

        # Relabel nodes to be consecutive integers starting from 1
        G = networkx.relabel.convert_node_labels_to_integers(G, first_label=1)

        # Remove self-loops
        self_loops = [(u, v) for u, v in G.edges() if u == v]
        G.remove_edges_from(self_loops)

        # Write the graph in METIS format
        with open(output_file, "w") as f:
            f.write(f"{graph_order} {graph_size}\n")
            for _, neighbors in sorted(G.adjacency()):
                sorted_neighbors = sorted(int(nei) for nei in neighbors.keys())
                f.write(f"{' '.join(map(str, sorted_neighbors))}\n")

if __name__ == "__main__":
    # Create a simple example graph
    G = networkx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    # Set parameters for the ReduMIS solver
    params = {"time_limit": 10}  # 10 seconds time limit

    # Initialize and solve the MIS problem
    solver = ReduMIS(G, params)
    solver.solve()
