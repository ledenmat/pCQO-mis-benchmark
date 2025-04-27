import os
from copy import deepcopy
import networkx as nx
import pickle
import pandas
from datetime import datetime
import logging
import tqdm

# from solvers.CPSAT_MIS import CPSATMIS
# from solvers.Gurobi_MIS import GurobiMIS
from solvers.pCQO_MIS import pCQOMIS_MGD

# from solvers.KaMIS import ReduMIS

logger = logging.getLogger(__name__)
logging.basicConfig(filename="benchmark.log", level=logging.INFO, style="{")

# Interval for saving solution checkpoints
SOLUTION_SAVE_INTERVAL = 1

#### GRAPH IMPORT ####

# List of directories containing graph data
graph_directories = [
    "./graphs/gnm_random_graph_scalability/gnm_2000_999500",
    # "./graphs/gnm_random_graph_scalability/gnm_1500_562125",
    # "./graphs/gnm_random_graph_scalability/gnm_1000_249750",
    # "./graphs/gnm_random_graph_scalability/gnm_500_62375",
    # "./graphs/gnm_random_graph_scalability/gnm_50_613"
]

# Initialize dataset and names lists
dataset = {}
dataset_names = []

# List to store file paths of graphs
graph_list = []

# Collect all .gpickle files from the specified directories
for graph_directory in graph_directories:
    for filename in os.listdir(graph_directory):
        if filename.endswith(".gpickle"):
            graph_list.append(os.path.join(graph_directory, filename))
            dataset_names.append({"name": os.path.join(graph_directory, filename)[:-8]})

# Sort graph list and dataset names by the length of their names
graph_list = sorted(graph_list, key=lambda x: len(x))
dataset_names = sorted(dataset_names, key=lambda x: len(x["name"]))

#### SOLVER DESCRIPTION ####

# Define solvers and their parameters
base_solvers = [
    {
        "name": "pCQO GNM 1500-2000 Scalability",
        "class": pCQOMIS_MGD,
        "params": {
            "learning_rate": 0.01,
            "momentum": 0.55,
            "number_of_total_steps": 10000,
            "number_of_steps_per_batch": 200,
            "gamma": 100,
            "gamma_prime": 10,
            "batch_size": 2048,
            "sampling_stddev": 1,
            "output_interval": 5,
        },
    },
    # {
    #     "name": "pCQO GNM 1500-2000 Scalability lr=0.009",
    #     "class": pCQOMIS_MGD,
    #     "params": {
    #         "learning_rate": 0.009,
    #         "momentum": 0.55,
    #         "number_of_total_steps": 10000,
    #         "number_of_steps_per_batch": 200,
    #         "gamma": 100,
    #         "gamma_prime": 10,
    #         "batch_size": 2048,
    #         "sampling_stddev": 1,
    #         "output_interval": 5,
    #     },
    # },
    # {
    #     "name": "pCQO GNM 50-1000 Scalability",
    #     "class": pCQOMIS_MGD,
    #     "params": {
    #         "learning_rate": 0.01,
    #         "momentum": 0.55,
    #         "number_of_total_steps": 10000,
    #         "number_of_steps_per_batch": 200,
    #         "gamma": 100,
    #         "gamma_prime": 5,
    #         "batch_size": 2048,
    #         "sampling_stddev": 1,
    #         "output_interval": 5,
    #     },
    # },
    # {"name": "Gurobi", "class": GurobiMIS, "params": {}},
    # {"name": "CPSAT", "class": CPSATMIS, "params": {}},
    # {"name": "ReduMIS", "class": ReduMIS, "params": {}},
]

# # List of solvers to be used in the benchmarking
solvers = base_solvers

# solvers = []
## Grid Search (Commented Out)
# Uncomment and configure the following section for hyperparameter tuning
# for solver in base_solvers:
#     for learning_rate in [0.01, 0.009]:
#         for momentum in [0.55]:
#             for steps_per_batch in [200]:
#                 for gamma_gamma_prime in [(100,10)]:
#                     modified_solver = deepcopy(solver)
#                     modified_solver["name"] = (
#                         f"{modified_solver['name']} learning_rate={learning_rate}, momentum={momentum}, steps_per_batch={steps_per_batch}, gamma={gamma_gamma_prime[0]}, gamma_prime={gamma_gamma_prime[1]}"
#                     )
#                     modified_solver["params"]["learning_rate"] = learning_rate
#                     modified_solver["params"]["momentum"] = momentum
#                     modified_solver["params"]["steps_per_batch"] = steps_per_batch
#                     modified_solver["params"]["gamma"] = gamma_gamma_prime[0]
#                     modified_solver["params"]["gamma_prime"] = gamma_gamma_prime[1]
#                     solvers.append(modified_solver)


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
            column_headings = [
                solution["solution_method"] for solution in dataset_solutions
            ]

            # Collect sizes and times for each solution
            table_row.extend(
                [solution["data"]["size"] for solution in dataset_solutions]
            )
            table_row.extend(
                [
                    solution["data"]["initializations_solved"]
                    for solution in dataset_solutions
                ]
            )
            table_row.extend([solution["time_taken"] for solution in dataset_solutions])

            table_data.append(table_row)

    # Generate headers for the CSV file
    table_headers = ["Dataset Name"]
    table_headers.extend([heading + " Solution Size" for heading in column_headings])
    # Uncomment to include headers for steps to solution size if available
    table_headers.extend(
        [heading + " # initializations_solved" for heading in column_headings]
    )
    table_headers.extend([heading + " Solution Time" for heading in column_headings])

    # Save the data to a CSV file
    table = pandas.DataFrame(table_data, columns=table_headers)
    table.to_csv(
        f"zero_to_stage_{current_stage}_of_{total_stages}_total_stages_{datetime.now()}.csv"
    )


#### BENCHMARKING CODE ####
solutions = []

# Calculate total number of stages
stage = 0
stages = len(solvers) * len(graph_list)

# Iterate over each graph file
for graph_filename in tqdm.tqdm(
    graph_list, desc=" Iterating Through Graphs", position=0
):
    print(f"Graph {graph_filename} is being imported...")
    with open(graph_filename, "rb") as f:
        G = pickle.load(f)
        dataset = {
            "name": graph_filename[:-8],
            "graph": nx.relabel.convert_node_labels_to_integers(G, first_label=0),
        }

    # Iterate over each solver
    for index, solver in enumerate(
        tqdm.tqdm(solvers, desc=" Iterating Solvers for Each Graph")
    ):
        solver_instance = solver["class"](dataset["graph"], solver["params"])

        # Solve the problem using the current solver
        solver_instance.solve()
        solution = {
            "solution_method": solver["name"],
            "dataset_name": dataset["name"],
            "data": deepcopy(solver_instance.solution),
            "time_taken": deepcopy(solver_instance.solution_time),
        }
        logging.info(
            "CSV: %s, %s, %s",
            dataset["name"],
            solution["data"]["size"],
            solution["time_taken"],
        )
        solutions.append(solution)
        del solver_instance

        # Update progress and save checkpoint if necessary
        stage += 1
        logger.info("Completed %s / %s", stage, stages)

        if stage % (SOLUTION_SAVE_INTERVAL * len(solvers)) == 0:
            logger.info("Now saving a checkpoint.")
            table_output(solutions, dataset_names, stage, stages)

    del dataset
    del G

# Save final results
logger.info("Now saving final results.")
table_output(solutions, dataset_names, stage, stages)
