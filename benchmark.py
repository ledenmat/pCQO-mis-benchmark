import os
from copy import deepcopy
import networkx as nx
import pickle
import pandas

import torch

from lib.dataset_generation import assemble_dataset_from_gpickle

from solvers.Quadratic_Batch import Quadratic_Batch
# from solvers.Quadratic import Quadratic
# from solvers.KaMIS import ReduMIS


#### GRAPH IMPORT ####

graph_directories = [
    "./graphs/satlib/m403",
    "./graphs/satlib/m411",
    "./graphs/satlib/m418",
    "./graphs/satlib/m423",
    "./graphs/satlib/m429",
    "./graphs/satlib/m435",
    "./graphs/satlib/m441",
    "./graphs/satlib/m449"
]

dataset = assemble_dataset_from_gpickle(graph_directories)

### for SATLIB bad instance testing:
# temp = []

# for i in [5,9,11,23,27,28,29,35,49,67]:
#     temp.append(datasets[i])

# datasets = temp

#### SOLVER DESCRIPTION ####

solvers = [
    # {
    #     "name": "ReduMIS Time Constrained",
    #     "class": ReduMIS,
    #     "params": {
    #         "seed": 13,
    #         "time_limit": 30
    #     }
    # }
    {
        "name": "Quadratic SDP Batch Test 1",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 537600,
            "gamma": 775,
            "batch_size": 1,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
    {
        "name": "Quadratic SDP Batch Test 16",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 33600,
            "gamma": 775,
            "batch_size": 16,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
        {
        "name": "Quadratic SDP Batch Test 32",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 16800,
            "gamma": 775,
            "batch_size": 32,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
            {
        "name": "Quadratic SDP Batch Test 64",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 8400,
            "gamma": 775,
            "batch_size": 64,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
                {
        "name": "Quadratic SDP Batch Test 128",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 4200,
            "gamma": 775,
            "batch_size": 128,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
                    {
        "name": "Quadratic SDP Batch Test 256",
        "class": Quadratic_Batch,
        "params": {
            "learning_rate": 0.9,
            "number_of_steps": 2100,
            "gamma": 775,
            "batch_size": 256,
            "std": 1.75,
            "threshold": 0.0,
        },
    },
    #     {
    #     "name": "Quadratic SDP Variance Test 0.5",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 0.5,
    #         "threshold": 0.0,
    #     },
    # },
    #         {
    #     "name": "Quadratic SDP Variance Test 0.75",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 0.75,
    #         "threshold": 0.0,
    #     },
    # },
    #             {
    #     "name": "Quadratic SDP Variance Test 1",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 1,
    #         "threshold": 0.0,
    #     },
    # },
    #                 {
    #     "name": "Quadratic SDP Variance Test 1.5",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 1.5,
    #         "threshold": 0.0,
    #     },
    # },
    #                     {
    #     "name": "Quadratic SDP Variance Test 2",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 2,
    #         "threshold": 0.0,
    #     },
    # },
    #                         {
    #     "name": "Quadratic SDP Variance Test 5",
    #     "class": Quadratic_Batch,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 128,
    #         "std": 5,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "Quadratic with SDP",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 200,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #     },
    # },
    # {
    #     "name": "G250LR0.9",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 2000,
    #         "gamma": 250,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "G500LR0.9",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 2000,
    #         "gamma": 500,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    #         {
    #     "name": "G750LR0.9",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 2000,
    #         "gamma": 750,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    #     {
    #     "name": "G250LR0.5",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 2000,
    #         "gamma": 250,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "G500LR0.5",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 2000,
    #         "gamma": 500,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    #         {
    #     "name": "G750LR0.5",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 2000,
    #         "gamma": 750,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "G250LR0.3",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.3,
    #         "number_of_steps": 2000,
    #         "gamma": 250,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "G500LR0.3",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.3,
    #         "number_of_steps": 2000,
    #         "gamma": 500,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    #         {
    #     "name": "G750LR0.3",
    #     "class": Quadratic_SDP,
    #     "params": {
    #         "learning_rate": 0.3,
    #         "number_of_steps": 2000,
    #         "gamma": 750,
    #         "batch_size": 1,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #     },
    # },
    # {
    #     "name": "Uniform",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #     },
    # },
    # {
    #     "name": "Normal (mean=0.5, std=0.1)",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #         "value_initializer": lambda input : torch.nn.init.trunc_normal_(input, mean=0.5, std=0.1, a=0, b=1)
    #     },
    # },
    # {
    #     "name": "Normal (mean=0.5, std=0.15)",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #         "value_initializer": lambda input : torch.nn.init.trunc_normal_(input, mean=0.5, std=0.15, a=0, b=1)
    #     },
    # },
    # {
    #     "name": "Normal (mean=0.5, std=0.25)",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #         "value_initializer": lambda input : torch.nn.init.trunc_normal_(input, mean=0.5, std=0.25, a=0, b=1)
    #     },
    # },
    # {
    #     "name": "Normal (mean=0.25, std=0.1)",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #         "value_initializer": lambda input : torch.nn.init.trunc_normal_(input, mean=0.25, std=0.1, a=0, b=1)
    #     },
    # },
    # {
    #     "name": "Normal (mean=0.75, std=0.1)",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.03,
    #         "value_initializer": lambda input : torch.nn.init.trunc_normal_(input, mean=0.75, std=0.1, a=0, b=1)
    #     },
    # },

    #     {
    #     "name": "Carl-Net",
    #     "class": DNNMIS,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 1000,
    #         "solve_interval": 100,
    #         "weight_decay": 0.05,
    #         "selection_criteria": 0.45
    #     },
    # },
    # {
    #     "name": "Quadratic Standard",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1000,
    #         "gamma": 2000,
    #         "batch_size": 128,
    #         "lr_gamma": 0.1,
    #         "threshold": 0.0,
    #         "normalize": True
    #     },
    # },
    #     {
    #     "name": "Quadratic Standard",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 3000,
    #         "gamma": 50,
    #         "batch_size": 128,
    #         "lr_gamma": 0.2
    #     },
    # },
    # {
    #     "name": "Quadratic Normalized",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 3000,
    #         "gamma": 775,
    #         "batch_size": 512,
    #         "lr_gamma": 0.2,
    #         "normalize": True
    #     },
    # },
    # {
    #     "name": "Quadratic Combined",
    #     "class": Quadratic,
    #     "params": {
    #         "learning_rate": 0.5,
    #         "number_of_steps": 6000,
    #         "gamma": 775,
    #         "batch_size": 512,
    #         "lr_gamma": 0.2,
    #         "combine": True
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

initializations = pickle.load(open("./solutions/SDP/SDP_Generation_SATLIB", "rb"))

for graph in dataset:
    for solver in solvers:
        solver_instance = solver["class"](graph["data"], solver["params"])
        solver_instance.value_initializer = lambda _ : torch.normal(
                        mean=initializations[graph["name"]]["SDP_solution"], std=torch.sqrt(torch.ones((len(initializations[graph["name"]]["SDP_solution"]))))*solver["params"]["std"]
                    )
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

        if stage % (10*len(solvers)) == 0:
            print("Now saving a check point.")
            table_output(solutions, dataset, stage, stages)

print("Now saving final results.")
table_output(solutions, dataset, stage, stages)
