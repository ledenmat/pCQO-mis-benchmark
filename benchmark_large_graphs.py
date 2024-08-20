import os
from copy import deepcopy
import networkx as nx
import pickle
import torch
import pandas

from solvers.CPSATMIS import CPSATMIS, GurobiMIS
from solvers.Quadratic_Batch import Quadratic_Batch
from solvers.KaMIS import ReduMIS

#### GRAPH IMPORT ####

graph_directories = ["./graphs/gnm_random_graph_800"]

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

og_solvers = [
    {
        "name": "Quadratic GNM Scalability",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.1,
            "number_of_steps": 20000,
            "gamma": 4000,
            "batch_size": 1024,
            "std": 2.25,
            "threshold": 0.0,
            "steps_per_batch": 1000,
            "graphs_per_optimizer": 256,
        },
    },
    # {"name": "Gurobi", "class": GurobiMIS, "params": {}},
    # {"name": "CPSAT", "class": CPSATMIS, "params": {}},
    # {
    #     "name": "ReduMIS",
    #     "class": ReduMIS,
    #     "params": {"time_limit": 100},
    # },
]

solvers = []

for solver in og_solvers:
    for beta in [(.1, .4)]:
        for steps_per_batch in [400]:
            for batch_size in [2048]:
                for lr in [.5]:
                    for number_of_steps in [10000]:
                        modified_solver = deepcopy(solver)
                        modified_solver["name"] = f"{modified_solver['name']} b1={beta[0]} b2={beta[1]} steps_per_batch={steps_per_batch} lr={lr} batch_size={batch_size} number_of_steps={number_of_steps}"
                        modified_solver["params"]["steps_per_batch"] = steps_per_batch
                        modified_solver["params"]["adam_beta_1"] = beta[0]
                        modified_solver["params"]["adam_beta_2"] = beta[1]
                        modified_solver["params"]["learning_rate"] = lr
                        modified_solver["params"]["batch_size"] = batch_size
                        modified_solver["params"]["number_of_steps"] = number_of_steps
                        solvers.append(modified_solver)

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

    table_headers = ["Dataset Name"]

    table_headers.extend([solver["name"] + " Solution Size" for solver in solvers])
    table_headers.extend([solver["name"] + " Solution Time" for solver in solvers])

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

        ### Degree Based Initialization (Optional) ###
        mean_vector = []
        std = []
        degrees = dict(dataset["graph"].degree())

        # Find the maximum degree
        max_degree = max(degrees.values())

        for _, degree in dataset["graph"].degree():
            degree_init = 1 - degree / max_degree
            mean_vector.append(degree_init)

        min_degree_initialization = max(mean_vector)

        for i in range(len(mean_vector)):
            mean_vector[i] = mean_vector[i] / min_degree_initialization

        solver_instance.value_initializer = lambda _ : torch.normal(
                        mean=torch.Tensor(mean_vector), std=solver["params"]["std"]
                    )

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

        if stage % len(solvers) == 0:
            print("Now saving a check point.")
            table_output(solutions, dataset_names, stage, stages)
    del dataset
    del G

print("Now saving final results.")
table_output(solutions, dataset_names, stage, stages)
