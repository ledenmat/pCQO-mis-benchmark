from ortools.sat.python import cp_model
import networkx as nx
import numpy as np
from lib.Solver import Solver
import time


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """
    A callback class for printing intermediate solutions during the search process.

    This class inherits from `CpSolverSolutionCallback` and is used to track and print
    intermediate solutions found by the CP-SAT solver.

    Attributes:
        _variables (list of cp_model.IntVar): List of variables representing nodes in the graph.
        _solution_count (int): Counter for the number of solutions found.
        _solution_limit (int): The limit on the number of solutions to print before stopping the search.
        start_time (float): Time when the search started.
        times (list of float): List to store the time taken to find each solution.
        paths (list of int): List to store the size of each solution.
    """
    
    def __init__(self, variables, limit):
        """
        Initializes the solution printer with variables and solution limit.

        Args:
            variables (list of cp_model.IntVar): The variables to track during the search.
            limit (int): The maximum number of solutions to print.
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._variables = variables
        self._solution_count = 0
        self._solution_limit = limit
        self.start_time = time.time()

        self.times = []
        self.paths = []

    def on_solution_callback(self):
        """
        Callback method that is called at each new solution found by the solver.

        This method records the solution size, the time taken, and prints the solution
        details. It also stops the search if the solution limit is reached.
        """
        solution_size = 0
        self.times.append(time.time() - self.start_time)
        for v in self._variables:
            solution_size += self.Value(v)

        self.paths.append(solution_size)

        self._solution_count += 1
        print(f'Solution {self._solution_count}:')
        # for v in self._variables:
        #     print(f'  {v.Name()} = {self.Value(v)}')
        if self._solution_count >= self._solution_limit:
            self.StopSearch()  # Optional: Stop search after N solutions

    def solution_count(self):
        """
        Returns the number of solutions found by the callback.

        Returns:
            int: The number of solutions found.
        """
        return self._solution_count


class CPSATMIS(Solver):
    """
    A solver class for finding the Maximum Independent Set (MIS) of a graph using
    the Google OR-Tools CP-SAT solver.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - time_limit (int, optional): Time limit (in seconds) for the solver to run. Defaults to None.
    """

    def __init__(self, G, params):
        """
        Initializes the CPSATMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver, including optional time_limit.
        """
        self.G = G
        self.time_limit = params.get("time_limit", None)
        self.solution = {}
        self.solution_time = None
        self.print_intermediate = True

        self.times = []
        self.paths = []

    def solve(self):
        """
        Executes the CP-SAT solver to find the Maximum Independent Set (MIS) of the graph.

        The method performs the following steps:
        1. Creates and configures a new CP-SAT model.
        2. Defines binary variables for each node in the graph.
        3. Adds constraints to ensure no two adjacent nodes are both in the independent set.
        4. Sets the objective to maximize the number of nodes in the independent set.
        5. Optionally prints intermediate solutions and their sizes.
        6. Solves the model and extracts the solution if found.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (numpy.ndarray): Array where 1s denote nodes in the MIS.
                - size (int): Size of the MIS.
        """
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        # Set time limit if specified
        if self.time_limit is not None:
            solver.parameters.max_time_in_seconds = float(self.time_limit)

        # Create binary variables for each node
        node_vars = {node: model.NewBoolVar(f"node_{node}") for node in self.G.nodes}

        # Add constraints: no two adjacent nodes can both be in the independent set
        for u, v in self.G.edges:
            model.Add(node_vars[u] + node_vars[v] <= 1)

        # Objective: Maximize the sum of the variables (maximize the size of the independent set)
        model.Maximize(sum(node_vars[node] for node in node_vars))

        if self.print_intermediate:
            # Prepare the solution printer
            solution_printer = VarArraySolutionPrinter(
                list(node_vars.values()), limit=30
            )
            # Start the solver and pass the solution printer
            status = solver.Solve(model, solution_printer)
            print(f"Number of solutions found: {solution_printer.solution_count()}")
        else:
            # Start the solver without the solution printer
            status = solver.Solve(model)

        # Check if a valid solution exists
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution_nodes = [
                node for node, var in node_vars.items() if solver.Value(var) == 1
            ]
            self.solution["graph_mask"] = np.zeros(len(node_vars))
            self.solution["graph_mask"][solution_nodes] = 1
            self.solution["size"] = len(solution_nodes)
        else:
            self.solution["graph_mask"] = np.zeros(len(node_vars))
            self.solution["size"] = 0

        self.solution_time = solver.WallTime()


if __name__ == "__main__":
    # Create a simple example graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    # Set parameters for the CPSATMIS solver
    params = {"time_limit": 10}  # 10 seconds time limit

    # Initialize and solve the MIS problem
    solver = CPSATMIS(G, params)
    solver.solve()
