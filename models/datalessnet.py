import torch
import torch.nn as nn
from torch import Tensor

import networkx as nx
from networkx import MultiGraph


class ConstrainedElemMultiply(nn.Module):
    def __init__(self, in_features, out_features, lower_bound=0, upper_bound=1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data = self.weight.data.clamp(self.lower_bound, self.upper_bound)
        return x * self.weight


def generate_theta_weight(graph: MultiGraph, graph_order: int) -> nn.Parameter:
    theta_weight = torch.zeros(graph_order)

    node_degrees = torch.tensor([val for (_, val) in graph.degree()])

    max_degree = node_degrees.max()

    for i in range(graph_order):
        theta_weight[i] = 1- node_degrees[i] / max_degree + torch.randint(0, 1, size=(1,)) /10

    # torch.manual_seed(7)

    # theta_weight = torch.rand((graph_order))

    return nn.Parameter(theta_weight)


def generate_second_layer_weight(
    graph: MultiGraph, graph_order: int, graph_size: int, graph_c_size: int
) -> nn.Parameter:
    second_layer_weight = torch.zeros(
        graph_order, graph_order + graph_size + graph_c_size
    )

    for i in range(graph_order):
        second_layer_weight[i, i] = 1.0

    for i, pair in enumerate(graph.edges):
        second_layer_weight[pair[0], graph_order + i] = 1.0
        second_layer_weight[pair[1], graph_order + i] = 1.0

    complement_graph: MultiGraph = nx.complement(graph)
    for i, pair in enumerate(complement_graph.edges):
        second_layer_weight[pair[0], graph_order + graph_size + i] = 1.0
        second_layer_weight[pair[1], graph_order + graph_size + i] = 1.0
    del complement_graph

    return nn.Parameter(second_layer_weight.t().to_dense())


def generate_second_layer_biases(
    graph_order: int, graph_size: int, graph_c_size: int
) -> nn.Parameter:
    second_layer_biases = torch.zeros(graph_order + graph_size + graph_c_size)

    second_layer_biases[0:graph_order] = -0.5
    second_layer_biases[graph_order:] = -1.0

    return nn.Parameter(second_layer_biases)


def generate_third_layer_weight(
    graph_order: int, graph_size: int, graph_c_size: int
) -> nn.Parameter:
    third_layer_weight = torch.zeros(graph_order + graph_size + graph_c_size)

    third_layer_weight[0:graph_order] = -1.0
    third_layer_weight[graph_order : graph_order + graph_size] = graph_order
    third_layer_weight[graph_order + graph_size :] = -1.0

    return nn.Parameter(third_layer_weight.to_dense())


class DatalessNet(nn.Module):
    def __init__(self, graph: MultiGraph) -> None:
        super().__init__()
        G = nx.relabel.convert_node_labels_to_integers(graph)
        self.graph_order = len(G.nodes)
        self.graph_size = len(G.edges)
        self.graph_c_size = len(nx.complement(G).edges)

        self.temperature = 0.5

        self.gamma = self.graph_order

        self.theta_layer = ConstrainedElemMultiply(
            in_features=self.graph_order, out_features=self.graph_order
        )
        with torch.no_grad():
            self.theta_layer.weight = generate_theta_weight(G, self.graph_order)

        self.layer2 = nn.Linear(
            in_features=self.graph_order,
            out_features=self.graph_order + self.graph_size + self.graph_c_size,
        )
        self.layer2.weight = generate_second_layer_weight(
            G, self.graph_order, self.graph_size, self.graph_c_size
        )
        self.layer2.bias = generate_second_layer_biases(
            self.graph_order, self.graph_size, self.graph_c_size
        )
        self.layer2.requires_grad_(False)

        self.layer3 = nn.Linear(
            in_features=self.graph_order + self.graph_size + self.graph_c_size,
            out_features=1,
            bias=False,
        )

        self.layer3.weight = generate_third_layer_weight(
            self.graph_order, self.graph_size, self.graph_c_size
        )
        self.layer3.requires_grad_(False)

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.layer3.weight.data[self.graph_order + self.graph_size :] = (
                -1.0 * self.temperature
            )
            self.layer2.bias.data[0 : self.graph_order] = -1.0 * self.temperature

        x = self.theta_layer(x)
        x = self.activation(self.layer2(x))
        x = self.layer3(x)

        return x

    def update_gamma(self) -> None:
        with torch.no_grad():
            self.layer3.weight.data[
                self.graph_order : self.graph_order + self.graph_size
            ] = self.gamma
