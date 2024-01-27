import os
from copy import deepcopy
import networkx as nx
import pickle
import pandas

from solvers.Quadratic import Quadratic

#### GRAPH IMPORT ####

graph_directories = ["./graphs/er_700-800/", "./graphs/satlib/"]

datasets = []


for filename in os.listdir(graph_directories[0]):
    if filename.endswith(".gpickle"):
        print("Graph ", os.path.join(graph_directories[0], filename), "is being imported ...")
        with open(os.path.join(graph_directories[0], filename), 'rb') as f:
            G = pickle.load(f)
            datasets.append(
            {
                "name": filename[:-8],
                "graph": nx.relabel.convert_node_labels_to_integers(
                    G, first_label=0
                ),
            })

datasets = datasets[2:3] + datasets[4:5]

for filename in os.listdir(graph_directories[1]):
    if filename.endswith(".gpickle"):
        print("Graph ", os.path.join(graph_directories[1], filename), "is being imported ...")
        with open(os.path.join(graph_directories[1], filename), 'rb') as f:
            G = pickle.load(f)
            datasets.append(
            {
                "name": filename[:-8],
                "graph": nx.relabel.convert_node_labels_to_integers(
                    G, first_label=0
                ),
            })

datasets = datasets[0:4]

#### SOLVER DESCRIPTION ####

solvers = [
    {
        "name": "Quadratic GPU",
        "class": Quadratic,
        "params": {
            "learning_rate": 0.05,
            "max_steps": 5000
        },
    }
]

solver = solvers[0]

#### BENCHMARKING CODE ####
solutions = []

betas = [0.25]
gammas = range(200, 2000, 25)
stage = 0
stages = len(betas) * len(gammas) * len(datasets)

best_IS = [0] * len(datasets)
best_gamma = [[0]] * len(datasets)
best_beta = [[0]] * len(datasets)

for beta in betas:
    for gamma in gammas:
        for i, dataset in enumerate(datasets):
            print(len(dataset["graph"]), len(dataset["graph"].edges()))
            solver_instance = solver["class"](dataset["graph"], solver["params"])
            solver_instance.gamma = gamma
            solver_instance.beta = beta
            solver_instance.solve()
            solution = {
                "solution_method": solver["name"],
                "dataset_name": dataset["name"],
                "data": deepcopy(solver_instance.solution),
                "time_taken": deepcopy(solver_instance.solution_time),
            }

            if best_IS[i] < int(solution['data']['size']):
                best_IS[i] = int(solution['data']['size'])
                best_beta[i] = [beta]
                best_gamma[i] = [gamma]
                print(f"Larger MIS found with {beta}, {gamma}, {solution['data']['size']}        CSV: {dataset['name']}, {solution['data']['size']}, {solution['time_taken']}")
            elif best_IS[i] == int(solution['data']['size']):
                best_beta[i].append(beta)
                best_gamma[i].append(gamma)
            else:
                print(f"sad - best gamma {best_gamma[i]}, best beta {best_beta[i]}, largest IS {best_IS[i]}")
            solutions.append(solution)
            del solver_instance
            stage += 1
            print(f"Completed {stage} / {stages}")

print("Now saving final results.")
print(best_IS)
print(best_beta)
print(best_gamma)