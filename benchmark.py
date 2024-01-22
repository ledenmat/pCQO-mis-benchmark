import os
from copy import deepcopy
import networkx as nx
import pickle
import pandas

from solvers.dNNMIS_GPU_TAU import DNNMIS

#### GRAPH IMPORT ####

graph_directories = ["./graphs/er_700-800/", "./graphs/satlib/"]

datasets = []

for graph_directory in graph_directories:
    for filename in os.listdir(graph_directory):
        if filename.endswith(".gpickle"):
            print("Graph ", os.path.join(graph_directory, filename), "is being imported ...")
            with open(os.path.join(graph_directory, filename), 'rb') as f:
                G = pickle.load(f)
                datasets.append(
                {
                    "name": filename[:-8],
                    "graph": nx.relabel.convert_node_labels_to_integers(
                        G, first_label=0
                    ),
                })

#### SOLVER DESCRIPTION ####

solvers = [
    {
        "name": "dNN GPU",
        "class": DNNMIS,
        "params": {
            "learning_rate": 0.05,
            "weight_decay": 0.05,
            "selection_criteria": 0.45,
            "max_steps": 100001
        },
    }
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
stages = len(solvers) * len(datasets)

for solver in solvers:
    for dataset in datasets:
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
            table_output(solutions, datasets, stage, stages)

print("Now saving final results.")
table_output(solutions, datasets, stage, stages)
