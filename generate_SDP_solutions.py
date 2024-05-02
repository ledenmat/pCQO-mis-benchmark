import os
import time
from copy import deepcopy
import networkx as nx
import pickle
import argparse

from lib.SDP import solve_SDP

SAVE_CHECKPOINT_EVERY = 50


#### GRAPH IMPORT ####

def assemble_dataset_from_gpickle(graph_directories):
    dataset = []
    for graph_directory in graph_directories:
        for filename in os.listdir(graph_directory):
            if filename.endswith(".gpickle"):
                print(
                    "Graph ",
                    os.path.join(graph_directory, filename),
                    "is being imported ...",
                )
                with open(os.path.join(graph_directory, filename), "rb") as f:
                    G = pickle.load(f)
                    dataset.append(
                        {
                            "name": filename[:-8],
                            "data": nx.relabel.convert_node_labels_to_integers(
                                G, first_label=0
                            ),
                        }
                    )
    return dataset


#### Save to Pickle ####
def save_gpickle(file_path, python_object):
    with open(file_path, 'wb') as file:
        pickle.dump(python_object, file)


def generate_SDP_solutions(dataset, output_directory, save_interval):
    solutions = []

    stage = 0
    stages = len(dataset)

    for graph in dataset:
        start_time = time.time()
        SDP_solution = solve_SDP(graph["data"])
        total_time = time.time() - start_time
        solution = {
            "graph_name": graph["name"],
            "SDP_solution": deepcopy(SDP_solution),
            "time_taken": deepcopy(total_time),
        }
        print(
            f"Graph Name: {solution['graph_name']}, SDP Solve Time Taken: {solution['time_taken']}"
        )
        solutions.append(solution)
        stage += 1
        print(f"Completed {stage} / {stages}")

        if stage % (save_interval) == 0:
            print("Now saving a check point.")
            save_gpickle(f"{output_directory}SDP_Generation_Stage-{stage}_of_{stages}", solutions)

    print("Now saving final results.")
    save_gpickle(f"{output_directory}SDP_Generation_{stages}", solutions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDP Solutions for a dataset of graphs.")
    parser.add_argument("checkpoint_freq", type=int, help="Save checkpoint every N number of steps")
    parser.add_argument("output_dir", type=str, help="Data output directory")
    parser.add_argument("input_dirs", nargs="+", type=str, help="Data input directories (places where you store graphs)")

    args = parser.parse_args()

    dataset = assemble_dataset_from_gpickle(args.input_dirs)

    generate_SDP_solutions(dataset, output_directory=args.output_dir, save_interval=args.checkpoint_freq)