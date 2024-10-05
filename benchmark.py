from copy import deepcopy
import pickle
import pandas
import torch
from datetime import datetime
import logging
import tqdm

from lib.dataset_generation import assemble_dataset_from_gpickle
from solvers.pCQO_MIS import pCQOMIS
from solvers.pCQO_MIS_GD import pCQOMIS_GD
from solvers.CPSAT_MIS import CPSATMIS
from solvers.Gurobi_MIS import GurobiMIS
from solvers.KaMIS import ReduMIS
from solvers.dNN_Alkhouri_MIS import DNNMIS

logger = logging.getLogger(__name__)
logging.basicConfig(filename='benchmark.log', level=logging.INFO, style="{")

# Interval for saving solution checkpoints
SOLUTION_SAVE_INTERVAL = 2

#### GRAPH IMPORT ####

# List of directories containing graph data
graph_directories = [
    ### ER 700-800 Graphs ###
    # "./graphs/er_700-800"
    ### GNM 300 Convergence Graphs ###
    # "./graphs/gnm_random_graph_convergence",
    ### SATLIB Graphs ###
    "./graphs/satlib/m403",
    "./graphs/satlib/m411",
    "./graphs/satlib/m418",
    "./graphs/satlib/m423",
    "./graphs/satlib/m429",
    "./graphs/satlib/m435",
    "./graphs/satlib/m441",
    "./graphs/satlib/m449",
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
base_solvers = [
    # {"name": "Gurobi", "class": GurobiMIS, "params": {"time_limit": 30}},
    # {"name": "CPSAT", "class": CPSATMIS, "params": {"time_limit": 30}},
    # {"name": "ReduMIS", "class": ReduMIS, "params": {}},
    # {
    #     "name": "pCQO_MIS ER",
    #     "class": pCQOMIS,
    #     "params": {
    #         "adam_beta_1": 0.1,
    #         "adam_beta_2": 0.25,
    #         "learning_rate": 0.6,
    #         "number_of_steps": 9900,
    #         "gamma": 775,
    #         "batch_size": 256,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 50,
    #         "graphs_per_optimizer": 256,
    #         "output_interval": 9900,
    #         "value_initializer": "degree"
    #     },
    # },
    #     {
    #     "name": "pCQO_MIS ER GD",
    #     "class": pCQOMIS_GD,
    #     "params": {
    #         "learning_rate": 0.000009,
    #         "momentum": 0.9,
    #         "number_of_steps": 9000,
    #         "gamma": 350,
    #         "beta": 7,
    #         "batch_size": 256,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 450,
    #         "output_interval": 25000,
    #         "value_initializer": "degree"
    #     },
    # },
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
    #         "value_initializer": "degree"
    #     },
    # }
    # {
    #     "name": "pCQO_MIS SATLIB modified",
    #     "class": pCQOMIS,
    #     "params": {
    #         "adam_beta_1": 0.9,
    #         "adam_beta_2": 0.99,
    #         "learning_rate": 0.9,
    #         "number_of_steps": 1500,
    #         "gamma": 775,
    #         "batch_size": 256,
    #         "std": 2.25,
    #         "threshold": 0.00,
    #         "steps_per_batch": 50,
    #         "output_interval": 10000,
    #         "value_initializer": "degree"
    #     },
    # }
    {
        "name": "pCQO_MIS SATLIB GD",
        "class": pCQOMIS_GD,
        "params": {
            "learning_rate": 0.0003,
            "momentum": 0.875,
            "number_of_steps": 3000,
            "gamma": 900,
            "batch_size": 256,
            "std": 2.25,
            "threshold": 0.00,
            "steps_per_batch": 50,
            "output_interval": 10000,
            "value_initializer": "degree"
        },
    }
]

# solvers = base_solvers

## Grid Search (Commented Out)
# Uncomment and configure the following section for hyperparameter tuning
# solvers = []
# for solver in base_solvers:
#     for learning_rate in [0.0003]:
#         for momentum in [0.875]:
#             for steps_per_batch in [30]:
#                 for gamma_beta in [(900, 1)]:
#                     for batch_size in [256]:
#                         for terms in ["three"]:
#                             modified_solver = deepcopy(solver)
#                             modified_solver["name"] = (
#                                 f"{modified_solver['name']} batch_size={batch_size}, learning_rate={learning_rate}, momentum={momentum}, steps_per_batch={steps_per_batch}, gamma={gamma_beta[0]}, beta={gamma_beta[1]},  terms={terms}"
#                             )
#                             modified_solver["params"]["learning_rate"] = learning_rate
#                             modified_solver["params"]["momentum"] = momentum
#                             modified_solver["params"]["steps_per_batch"] = steps_per_batch
#                             modified_solver["params"]["gamma"] = gamma_beta[0]
#                             modified_solver["params"]["beta"] = gamma_beta[1]
#                             modified_solver["params"]["number_of_terms"] = terms
#                             modified_solver["params"]["batch_size"] = batch_size
#                             solvers.append(modified_solver)

# solvers = []
# for solver in base_solvers:
#     for terms in ["two","three"]:
#         modified_solver = deepcopy(solver)
#         modified_solver["name"] = (
#             f"{modified_solver['name']} terms={terms}"
#         )
#         modified_solver["params"]["number_of_terms"] = terms
#         solvers.append(modified_solver)


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
    table.to_csv(f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages_{datetime.now()}.csv")

#### BENCHMARKING CODE ####
solutions = []

# Calculate total number of stages
stage = 0
stages = len(solvers) * len(dataset)

# Optional: Load SDP initializer if needed
# initializations = pickle.load(open("./solutions/SDP/SDP_Generation_SATLIB", "rb"))

# Iterate over each graph in the dataset
for graph in tqdm.tqdm(dataset, desc=" Iterating Through Graphs", position=0):
    for solver in tqdm.tqdm(solvers, desc=" Iterating Solvers for Each Graph"):
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
        logging.info("CSV: %s, %s, %s", graph['name'], solution['data']['size'], solution['time_taken'])
        solutions.append(solution)
        del solver_instance

        # Update progress and save checkpoint if necessary
        stage += 1
        logger.info("Completed %s / %s", stage, stages)


        if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
            logger.info("Now saving a checkpoint.")
            table_output(solutions, dataset, stage, stages)

# Save final results
logger.info("Now saving final results.")
table_output(solutions, dataset, stage, stages)
