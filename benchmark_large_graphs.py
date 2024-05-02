import os
from copy import deepcopy
import networkx as nx
import pickle
import pandas

from solvers.dNNMIS_GPU_TAU import DNNMIS
from solvers.Quadratic import Quadratic

#### GRAPH IMPORT ####

graph_directories = ["./graphs/gnm_random_graph_50-2000"]

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

solvers = [
    # {
    #     "name": "Quadratic Normalized",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.05,
    #         "number_of_steps": 2000,
    #         "gamma": 2000,
    #         "batch_size": 8,
    #         "lr_gamma": 0.2,
    #         "threshold": 0.0,
    #         "normalized": True
    #     },
    # },
    # {
    #     "name": "Quadratic Standard",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.05,
    #         "number_of_steps": 10000,
    #         "gamma": 2000,
    #         "batch_size": 8,
    #         "lr_gamma": 0.2,
    #         "threshold": 0.0,
    #         "normalized": False
    #     },
    # },
    {
        "name": "Carl-Net",
        "class": DNNMIS,
        "params": {
            "learning_rate": 0.05,
            "number_of_steps": 50000,
            "solve_interval": 100,
            "weight_decay": 0.05
        },
    },
]

#### SOLUTION OUTPUT FUNCTION ####
def table_output(solutions, datasets, current_stage, total_stages):
    dataset_index = {k: v for v, k in enumerate([dataset["name"] for dataset in datasets])}
    datasets_solutions = [[] for i in range(len(datasets))]
    table_data = []

    for solution in solutions:
        dsi = dataset_index[solution["dataset_name"]]
        datasets_solutions[dsi].append(solution)

    i = 0
    for dataset_solutions in datasets_solutions:
        if len(dataset_solutions) > 0:
            table_row = [dataset_solutions[0]['dataset_name']]

            table_row.extend([solution["data"]["size"] for solution in dataset_solutions])
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
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
        dataset = {
            "name": graph_filename[:-8],
            "graph": nx.relabel.convert_node_labels_to_integers(
                G, first_label=0
            ),
        }
    for solver in solvers:
        
        solver_instance = solver["class"](dataset["graph"], solver["params"])
        solver_instance.solve()
        solution = {
            "solution_method": solver["name"],
            "dataset_name": dataset["name"],
            "data": deepcopy(solver_instance.solution),
            "time_taken": deepcopy(solver_instance.solution_time),
        }
        print(f"CSV: {dataset['name']}, {solution['data']['size']}, {solution['time_taken']}")
        solutions.append(solution)
        del solver_instance
        stage += 1
        print(f"Completed {stage} / {stages}")

        if stage % 5 == 0:
            print("Now saving a check point.")
            table_output(solutions, dataset_names, stage, stages)
    del dataset
    del G

print("Now saving final results.")
table_output(solutions, dataset_names, stage, stages)
