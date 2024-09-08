from copy import deepcopy
import pickle
import pandas
import torch

from lib.dataset_generation import assemble_dataset_from_gpickle

from solvers.pCQO_MIS import pCQOMIS
from solvers.CPSAT_MIS import CPSATMIS
from solvers.Gurobi_MIS import GurobiMIS
from solvers.KaMIS import ReduMIS
from solvers.dNN_Alkhouri_MIS import DNNMIS

SOLUTION_SAVE_INTERVAL = 3


#### GRAPH IMPORT ####

graph_directories = [
    ### ER 700-800 Graphs ###
    "./graphs/er_700-800"
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
    # "./graphs/satlib/m449",
    ### ER density test Graphs ###
    # "./graphs/er_05",
    # "./graphs/er_10",
    # "./graphs/er_15",
    # "./graphs/er_20"
]

dataset = assemble_dataset_from_gpickle(graph_directories)

#### SOLVER DESCRIPTION ####

solvers = [
    {"name": "Gurobi", "class": GurobiMIS, "params": {"time_limit": 30}},
    {"name": "CPSAT", "class": CPSATMIS, "params": {"time_limit": 30}},
    # {"name": "ReduMIS", "class": ReduMIS, "params": {}},
    {
        "name": "pCQO_MIS ER",
        "class": pCQOMIS,
        "params": {
            "adam_beta_1": 0.1,
            "adam_beta_2": 0.25,
            "learning_rate": 0.6,
            "number_of_steps": 9900,
            "gamma": 775,
            "batch_size": 256,
            "std": 2.25,
            "threshold": 0.00,
            "steps_per_batch": 150,
            "graphs_per_optimizer": 256,
            "output_interval": 9900,
        },
    },
    # {
    #     "name": "pCQO_MIS SATLIB",
    #     "class": pCQOMIS,
    #     "params": {
    #         "adam_beta_1": 0.9,
    #         "adam_beta_2": 0.99,
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 256,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 50,
    #         "output_interval": 10000,
    #     },
    # }
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
            # table_row.extend([solution['data']['steps_to_best_MIS'] for solution in dataset_solutions])
            table_row.extend([solution["time_taken"] for solution in dataset_solutions])

            table_data.append(table_row)

    table_headers = ["Dataset Name"]

    table_headers.extend([heading + " Solution Size" for heading in column_headings])
    # table_headers.extend([heading + " # Steps to Solution Size" for heading in column_headings])
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
