import os
from copy import deepcopy
import networkx as nx
import pickle
import torch
import pandas

from solvers.CPSAT_MIS import CPSATMIS
from solvers.Gurobi_MIS import GurobiMIS
from solvers.pCQO_MIS import pCQOMIS
from solvers.KaMIS import ReduMIS

SOLUTION_SAVE_INTERVAL = 1

#### GRAPH IMPORT ####

graph_directories = ["./graphs/gnm_random_graph_scalability"]

dataset = {}

dataset_names = []

graph_list = []

for graph_directory in graph_directories:
    for filename in os.listdir(graph_directory):
        if filename.endswith(".gpickle"):
            graph_list.append(os.path.join(graph_directory, filename))
            dataset_names.append({"name": os.path.join(graph_directory, filename)[:-8]})

graph_list = sorted(graph_list, key=lambda x: len(x))
dataset_names = sorted(dataset_names, key=lambda x: len(x["name"]))

#### SOLVER DESCRIPTION ####

base_solvers = [
    {
        "name": "pCQO GNM 50-1500 Scalability",
        "class": pCQOMIS,
        "params": {
            "std": 2.25,
            "adam_beta_1": 0.1,
            "adam_beta_2": 0.4,
            "learning_rate": 0.5,
            "number_of_steps": 100000,
            "batch_size": 2048,
            "steps_per_batch": 400,
        },
    },
    {"name": "Gurobi", "class": GurobiMIS, "params": {}},
    {"name": "CPSAT", "class": CPSATMIS, "params": {}},
    {"name": "ReduMIS", "class": ReduMIS, "params": {}},
]

solvers = base_solvers

## Grid Search
# for solver in base_solvers:
#     for number_of_steps in []:
#         for learning_rate in []:
#             for adam_beta in []:
#                 modified_solver = deepcopy(solver)
#                 modified_solver["name"] = (
#                     f"{modified_solver['name']} learning_rate={learning_rate} adam_beta={adam_beta} number_of_steps={number_of_steps}"
#                 )
#                 modified_solver["params"][
#                     "learning_rate"
#                 ] = learning_rate
#                 modified_solver["params"]["adam_beta_1"] = adam_beta[0]
#                 modified_solver["params"]["adam_beta_2"] = adam_beta[1]
#                 modified_solver["params"][
#                     "number_of_steps"
#                 ] = number_of_steps
#                 modified_solver["params"]["output_interval"] = 1000
#                 solvers.append(modified_solver)


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

            table_row.extend(
                [solution["data"]["size"] for solution in dataset_solutions]
            )
            table_row.extend([solution["time_taken"] for solution in dataset_solutions])

            table_data.append(table_row)
            valid_dataset_solutions = dataset_solutions

    solution_size_header = [solver["name"] + " Solution Size" for solver in solvers]
    solution_time_header = [solver["name"] + " Solution Time" for solver in solvers]

    solution_size_header = solution_size_header[: len(valid_dataset_solutions)]
    solution_time_header = solution_time_header[: len(valid_dataset_solutions)]

    table_headers = ["Dataset Name"]

    table_headers.extend(solution_size_header)
    table_headers.extend(solution_time_header)

    table = pandas.DataFrame(table_data, columns=table_headers)
    table.to_csv(f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages.csv")


#### BENCHMARKING CODE ####
solutions = []

stage = 0
stages = len(solvers) * len(graph_list)

for graph_filename in graph_list:
    print("Graph ", graph_filename, "is being imported ...")
    with open(graph_filename, "rb") as f:
        G = pickle.load(f)
        dataset = {
            "name": graph_filename[:-8],
            "graph": nx.relabel.convert_node_labels_to_integers(G, first_label=0),
        }
    for index, solver in enumerate(solvers):

        solver_instance = solver["class"](dataset["graph"], solver["params"])

        solver_instance.solve()
        solution = {
            "solution_method": solver["name"],
            "dataset_name": dataset["name"],
            "data": deepcopy(solver_instance.solution),
            "time_taken": deepcopy(solver_instance.solution_time),
        }
        print(
            f"CSV: {dataset['name']}, {solution['data']['size']}, {solution['time_taken']}"
        )
        solutions.append(solution)
        del solver_instance
        stage += 1
        print(f"Completed {stage} / {stages}")

        if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
            print("Now saving a check point.")
            table_output(solutions, dataset, stage, stages)
    del dataset
    del G

print("Now saving final results.")
table_output(solutions, dataset_names, stage, stages)
