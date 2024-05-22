from copy import deepcopy
import pickle
import pandas
import torch

from lib.dataset_generation import assemble_dataset_from_gpickle

from solvers.Quadratic_Batch import Quadratic_Batch
from solvers.CPSATMIS import CPSATMIS, GurobiMIS
from solvers.KaMIS import ReduMIS

SOLUTION_SAVE_INTERVAL = 5


#### GRAPH IMPORT ####

graph_directories = [
    ### ER 700-800 Graphs ###
    # "./graphs/er_700-800"
    ### GNM 300 Convergence Graphs ###
    # "./graphs/gnm_random_graph_convergence",
    ### SATLIB Graphs ###
    # "./graphs/satlib/m403",
    # "./graphs/satlib/m411",
    # "./graphs/satlib/m418",
    # "./graphs/satlib/m423",
    # "./graphs/satlib/m429",
    # "./graphs/satlib/m435",
    # "./graphs/satlib/m441",
    # "./graphs/satlib/m449"
]

dataset = assemble_dataset_from_gpickle(graph_directories)

#### SOLVER DESCRIPTION ####

solvers = [
    # {"name": "Gurobi", "class": GurobiMIS, "params": {"time_limit": 100}},
    # {"name": "CPSAT", "class": CPSATMIS, "params": {"time_limit": 30}},
    # {
    #     "name": "ReduMIS",
    #     "class": ReduMIS,
    #     "params": {"time_limit": 30},
    # },
    # {
    #     "name": "Quadratic SATLIB",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 4000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 100,
    #     },
    # },
    # {
    #     "name": "Quadratic ER",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.6,
    #         "number_of_steps": 9800,
    #         "gamma": 775,
    #         "batch_size": 256,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 350,
    #         "graphs_per_optimizer": 256,
    #     },
    # },
    # {
    #     "name": "Quadratic GNM Convergence",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.1,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 1024,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 300,
    #         "graphs_per_optimizer": 256,
    #     },
    # },
]


#### SOLUTION OUTPUT FUNCTION ####
def table_output(solutions, datasets, current_stage, total_stages):
    dataset_index = {
        k: v for v, k in enumerate([dataset["name"] for dataset in datasets])
    }
    datasets_solutions = [[] for i in range(len(datasets))]
    table_data = []

    for solution in solutions:
        dsi = dataset_index[solution["dataset_name"]]
        datasets_solutions[dsi].append(solution)

    i = 0
    for dataset_solutions in datasets_solutions:
        if len(dataset_solutions) > 0:
            table_row = [dataset_solutions[0]["dataset_name"]]

            column_headings = [
                solution["solution_method"] for solution in dataset_solutions
            ]

            table_row.extend(
                [solution["data"]["size"] for solution in dataset_solutions]
            )
            table_row.extend([solution["time_taken"] for solution in dataset_solutions])

            table_data.append(table_row)

    table_headers = ["Dataset Name"]

    table_headers.extend([heading + " Solution Size" for heading in column_headings])
    table_headers.extend([heading + " Solution Time" for heading in column_headings])

    table = pandas.DataFrame(table_data, columns=table_headers)
    table.to_csv(f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages.csv")


#### BENCHMARKING CODE ####
solutions = []

stage = 0
stages = len(solvers) * len(dataset)

### Part of SDP initializer (Optional) ###
# initializations = pickle.load(open("./solutions/SDP/SDP_Generation_SATLIB", "rb"))

for graph in dataset:
    for solver in solvers:
        solver_instance = solver["class"](graph["data"], solver["params"])

        ### Default initializer for Quadratic NN is Uniform Distribution

        ### Degree Based Initializer (Optional) ###
        # mean_vector = []
        # degrees = dict(graph["data"].degree())

        # # Find the maximum degree
        # max_degree = max(degrees.values())

        # for _, degree in graph["data"].degree():
        #     degree_init = 1 - degree / max_degree
        #     mean_vector.append(degree_init)

        # min_degree_initialization = max(mean_vector)

        # for i in range(len(mean_vector)):
        #     mean_vector[i] = mean_vector[i] / min_degree_initialization

        # solver_instance.value_initializer = lambda _: torch.normal(
        #     mean=torch.Tensor(mean_vector), std=solver["params"]["std"]
        # )
        ### End of Degree Based Initializer ###




        ### SDP Based Initializer (Optional) ###
        # solver_instance.value_initializer = lambda _ : torch.normal(
        #                 mean=initializations[graph["name"]]["SDP_solution"], std=torch.sqrt(torch.ones((len(initializations[graph["name"]]["SDP_solution"]))))*solver["params"]["std"]
        #             )
        ### End of SDP based Initializer ###



        solver_instance.solve()
        solution = {
            "solution_method": solver["name"],
            "dataset_name": graph["name"],
            "data": deepcopy(solver_instance.solution),
            "time_taken": deepcopy(solver_instance.solution_time),
        }
        print(
            f"CSV: {graph['name']}, {solution['data']['size']}, {solution['time_taken']}"
        )
        solutions.append(solution)
        del solver_instance
        stage += 1
        print(f"Completed {stage} / {stages}")

        if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
            print("Now saving a check point.")
            table_output(solutions, dataset, stage, stages)

print("Now saving final results.")
table_output(solutions, dataset, stage, stages)
