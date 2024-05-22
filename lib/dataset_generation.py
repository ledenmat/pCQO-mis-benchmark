import os
import pickle
import networkx as nx


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
