import os
import subprocess
from lib.Solver import Solver
import time
from pathlib import Path
import numpy
import networkx

module_directory = os.path.dirname(os.path.abspath(__file__))

# PARAMS:

## seed: What seed should be used for randomization?
## time_limit: How long should the algorithm run for?
## redumis_path: What command needs to be entered to use ReduMIS?

# OUTPUTS:

## solution:
### graph_mask: numpy array filled with 0s and 1s. 1s signify nodes in the MIS. Use this as a mask on the original graph to see MIS solution
### size: size of MIS


class ReduMIS(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        self.seed = params.get("seed", None)
        self.time_limit = params.get("time_limit", None)
        self.redumis_path = params.get(
            "redumis_path", os.path.join(module_directory, "../external/redumis")
        )
        self.solution = {}

    def solve(self):
        temp_graph_path = f"./temp_kamis_metis_{time.time()}"
        temp_graph_os_path = Path(temp_graph_path)

        temp_result_path = f"./temp_kamis_solution_{time.time()}"
        temp_result_os_path = Path(temp_result_path)

        self.networkx_to_metis(self.G, temp_graph_path)

        self._start_timer()

        redumis_command = [
            self.redumis_path,
            temp_graph_path,
            f"--output={temp_result_path}",
        ]

        if self.time_limit is not None:
            redumis_command.append(f"--time_limit={self.time_limit}")

        if self.seed is not None:
            redumis_command.append(f"--seed={self.seed}")

        subprocess.Popen(redumis_command, stdout=subprocess.PIPE, stderr=subprocess. STDOUT, shell=True)

        self._stop_timer()

        temp_graph_os_path.unlink(missing_ok=True)

        with open(temp_result_path, "r") as f:
            result = f.read().split("\n")
            result.pop()
            self.solution["graph_mask"] = numpy.array(result, dtype=int)
            self.solution["size"] = numpy.count_nonzero(
                self.solution["graph_mask"] == 1  
            )

        temp_result_os_path.unlink(missing_ok=True)

    @staticmethod
    def networkx_to_metis(G, output_file):
        graph_order = len((G.nodes))
        graph_size = len((G.edges))

        G = networkx.relabel.convert_node_labels_to_integers(G, first_label=1)

        self_loops = [(u, v) for u, v in G.edges() if u == v]
        G.remove_edges_from(self_loops)

        with open(output_file, "w") as f:
            f.write(f"{graph_order} {graph_size}\n")

            for _, neighbors in sorted(G.adjacency()):
                sorted_neighbors = list(map(int, neighbors.keys()))
                sorted_neighbors.sort()
                f.write(f"{' '.join(list(map(str, sorted_neighbors)))}\n")
