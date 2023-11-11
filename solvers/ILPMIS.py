from lib.Solver import Solver
import numpy as np
import cplex

# OUTPUTS:

## graph_mask: numpy array filled with 0s and 1s. 1s signify nodes in the MIS. Use this as a mask on the original graph to see MIS solution
## size: size of MIS


class ILPMIS(Solver):
    def __init__(self, G, params):
        super().__init__()
        self.G = G
        _ = params
        self.solution = {}

    def solve(self):
        G = self.G

        self._start_timer()
        ################################# remove self loops

        self_loop_removal_cntr = 0

        for pair in list(G.edges):
            if pair[0] == pair[1]:
                self.G.remove_edge(pair[0], pair[1])
                self_loop_removal_cntr = self_loop_removal_cntr + 1

        print(
            "ALREADY REMOVED {} EDGES BECAUSE OF SELF LOOP".format(
                self_loop_removal_cntr
            )
        )

        problem = cplex.Cplex()

        list_of_nodes = list(G.nodes)

        list_of_pairs_of_edges = list(G.edges)

        ### dictionary of id's

        node_id = {(n): "node_id({0})".format(n) for (n) in list_of_nodes}

        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(
            names=list(node_id.values()),
            lb=[0.0] * len(node_id),
            ub=[1.0] * len(node_id),
        )

        problem.variables.set_types(
            [(i, problem.variables.type.binary) for i in node_id.values()]
        )

        ##   objective:

        problem.objective.set_linear(
            list(zip(list(node_id.values()), [1.0] * len(node_id)))
        )

        ## constraint: for all (u,v)\in E, node_id(u) + node_id(v) <= 1

        """ Constraint (1) """

        for u, v in list_of_pairs_of_edges:
            # if u != v:

            lin_expr_vars_1 = []

            lin_expr_vals_1 = []

            lin_expr_vars_2 = []

            lin_expr_vals_2 = []

            lin_expr_vars_1.append(node_id[(u)])

            lin_expr_vals_1.append(1.0)

            lin_expr_vars_2.append(node_id[(v)])

            lin_expr_vals_2.append(1.0)

            problem.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        lin_expr_vars_1 + lin_expr_vars_2,
                        val=lin_expr_vals_2 + lin_expr_vals_2,
                    )
                ],
                rhs=[1.0],
                senses=["L"],
                names=["(1)_"],
            )

        problem.solve()

        self._stop_timer()

        if problem.solution.get_solution_type() == 0:
            print("CPLEX is outputting no solution exists")

        if problem.solution.get_solution_type() != 0:
            node_id_star = problem.solution.get_values()

            ### removing nodes in (node_id_star == 1) along with their nieghbors

            nodes_tobe_removed = np.where(np.array(node_id_star) == 1)

            if len(nodes_tobe_removed[0]) == 0:
                print(
                    "############# ILP is solved , BUT without ones. This scenario should nlt happen ##############"
                )

            MIS = nodes_tobe_removed[0]

            solution_graph = np.zeros(len(list_of_nodes))

            for node_id in MIS:
                solution_graph[node_id] = 1

            self.solution["graph_mask"] = solution_graph
            self.solution["size"] = len(MIS)
