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

# Interval for saving solution checkpoints
SOLUTION_SAVE_INTERVAL = 3

#### GRAPH IMPORT ####

# List of directories containing graph data
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

# Assemble dataset from .gpickle files in the specified directories
dataset = assemble_dataset_from_gpickle(graph_directories)

#### SOLVER DESCRIPTION ####

# Define solvers and their parameters
solvers = [
    {"name": "Gurobi", "class": GurobiMIS, "params": {"time_limit": 30}},
    {"name": "CPSAT", "class": CPSATMIS, "params": {"time_limit": 30}},
    {"name": "ReduMIS", "class": ReduMIS, "params": {}},
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
    # Uncomment and configure the following solver for SATLIB datasets if needed
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
    """
    Saves the solutions to a CSV file.

    Args:
        solutions (list): List of solution dictionaries.
        datasets (list): List of dataset dictionaries.
        current_stage (int): Current stage in the benchmarking process.
        total_stages (int): Total number of stages in the benchmarking process.
    """
    # Create a mapping of dataset names to indices
    dataset_index = {dataset["name"]: index for index, dataset in enumerate(datasets)}
    datasets_solutions = [[] for _ in range(len(datasets))]
    table_data = []

    # Organize solutions by dataset
    for solution in solutions:
        dataset_idx = dataset_index[solution["dataset_name"]]
        datasets_solutions[dataset_idx].append(solution)

    # Prepare data for the output table
    for dataset_solutions in datasets_solutions:
        if dataset_solutions:
            table_row = [dataset_solutions[0]["dataset_name"]]
            column_headings = [solution["solution_method"] for solution in dataset_solutions]

            # Collect sizes and times for each solution
            table_row.extend([solution["data"]["size"] for solution in dataset_solutions])
            # Uncomment to include steps to solution size if available
            # table_row.extend([solution['data']['steps_to_best_MIS'] for solution in dataset_solutions])
            table_row.extend([solution["time_taken"] for solution in dataset_solutions])

            table_data.append(table_row)

    # Generate headers for the CSV file
    table_headers = ["Dataset Name"]
    table_headers.extend([heading + " Solution Size" for heading in column_headings])
    # Uncomment to include headers for steps to solution size if available
    # table_headers.extend([heading + " # Steps to Solution Size" for heading in column_headings])
    table_headers.extend([heading + " Solution Time" for heading in column_headings])

    # Save the data to a CSV file
    table = pandas.DataFrame(table_data, columns=table_headers)
    table.to_csv(f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages.csv")

#### BENCHMARKING CODE ####
solutions = []

# Calculate total number of stages
stage = 0
stages = len(solvers) * len(dataset)

# Optional: Load SDP initializer if needed
# initializations = pickle.load(open("./solutions/SDP/SDP_Generation_SATLIB", "rb"))

# Iterate over each graph in the dataset
for graph in dataset:
    for solver in solvers:
        solver_instance = solver["class"](graph["data"], solver["params"])

        # Optional: Apply SDP-based initializer if needed
        # solver_instance.value_initializer = lambda _: torch.normal(
        #     mean=initializations[graph["name"]]["SDP_solution"],
        #     std=torch.sqrt(torch.ones((len(initializations[graph["name"]]["SDP_solution"])))) * solver["params"]["std"]
        # )

        # Solve the problem using the current solver
        solver_instance.solve()
        solution = {
            "solution_method": solver["name"],
            "dataset_name": graph["name"],
            "data": deepcopy(solver_instance.solution),
            "time_taken": deepcopy(solver_instance.solution_time),
        }
        print(f"CSV: {graph['name']}, {solution['data']['size']}, {solution['time_taken']}")
        solutions.append(solution)
        del solver_instance

        # Update progress and save checkpoint if necessary
        stage += 1
        print(f"Completed {stage} / {stages}")

        if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
            print("Now saving a checkpoint.")
            table_output(solutions, dataset, stage, stages)

# Save final results
print("Now saving final results.")
table_output(solutions, dataset, stage, stages)
