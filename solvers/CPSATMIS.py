from ortools.sat.python import cp_model
import networkx as nx
from gurobipy import Model, GRB, quicksum
import numpy as np
from lib.Solver import Solver
import csv
import time


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._variables = variables
        self._solution_count = 0
        self._solution_limit = limit
        self.start_time = time.time()

        self.times = []
        self.paths = []

    def on_solution_callback(self):
        """This method is called at each new solution."""
        solution_size = 0
        self.times.append(time.time() - self.start_time)
        for v in self._variables:
            solution_size += self.Value(v)

        self.paths.append(solution_size)

        print(self.paths, self.times)

        # self._solution_count += 1
        # print(f'Solution {self._solution_count}:')
        # for v in self._variables:
        #     print(f'  {v.Name()} = {self.Value(v)}')
        # if self._solution_count >= self._solution_limit:
        #     self.StopSearch()  # Optional: Stop search after N solutions

    def solution_count(self):
        return self._solution_count


class CPSATMIS(Solver):
    def __init__(self, G, params):
        self.G = G
        self.time_limit = params.get("time_limit", None)
        self.solution = {}
        self.solution_time = None
        self.print_intermediate = True

        self.times = []
        self.paths = []

    def solve(self):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        if self.time_limit is not None:
            solver.parameters.max_time_in_seconds = float(self.time_limit)

        # Create a binary variable for each node
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
            # Adjust limit as needed
            # Start the solver and pass the solution printer
            status = solver.Solve(model, solution_printer)
            print(f"Number of solutions found: {solution_printer.solution_count()}")
        else:
            # Start the solver without the solution printer
            status = solver.Solve(model)

        # Solve the model
        # status = solver.Solve(model)

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


class GurobiMIS(Solver):
    def __init__(self, G, params):
        self.G = G
        self.time_limit = params.get("time_limit", None)
        self.solution = {}
        self.model = None
        self.solution_time = None  # Initialize solution_time
        self.paths = []
        self.times = []

    def data_cb(self, model, where):
        if where == GRB.Callback.MIP:
            cur_obj = model.cbGet(GRB.Callback.MIP_OBJBST)

            self._stop_timer()
            if len(self.paths) == 0 or self.paths[-1] < cur_obj:
                self.times.append(self.solution_time)
                self.paths.append(cur_obj)

    def solve(self):
        # Create a new Gurobi model
        self.model = Model("Maximum_Independent_Set")

        # Set the time limit if specified
        if self.time_limit is not None:
            self.model.setParam("TimeLimit", self.time_limit)

        # Create a binary variable for each node
        node_vars = {
            node: self.model.addVar(vtype=GRB.BINARY, name=f"node_{node}")
            for node in self.G.nodes
        }

        # Add constraints: no two adjacent nodes can both be in the independent set
        for u, v in self.G.edges:
            if u != v:  # avoid adding a constraint for self-loops, if they exist
                self.model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")

        # Set the objective: maximize the sum of the selected nodes
        self.model.setObjective(
            quicksum(node_vars[node] for node in self.G.nodes), GRB.MAXIMIZE
        )

        # Optimize the model
        self._start_timer()
        self.model.optimize(callback=self.data_cb)
        self.solution_time = self.model.Runtime
        # Check if a valid solution exists
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            self.solution["graph_mask"] = [
                int(node_vars[node].X) for node in self.G.nodes
            ]
            self.solution["size"] = sum(self.solution["graph_mask"])
            print(f"Maximum Independent Set size: {self.solution['size']}")
        else:
            print("No valid solution found.")
            self.solution["graph_mask"] = []
            self.solution["size"] = 0

        print(self.paths, self.times)

        # Optional: Output the variables if the solution was found
        if self.model.status == GRB.OPTIMAL:
            print("Nodes in the independent set:")
            for node in self.G.nodes:
                if node_vars[node].X > 0.5:  # effectively checking if the variable is 1
                    print(node)


if __name__ == "__main__":
    # Create a simple example graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    # Set parameters
    params = {"time_limit": 10}  # 10 seconds time limit

    # Initialize and solve the MIS problem
    solver = GurobiMIS(G, params)
    solver.solve()
