import torch
import numpy
from custom_layers import ElementwiseMultiply

def datalessNN_module(theta_tensor, layer2_weights, layer2_biases, layer3_weights, torch_dtype=torch.float32):

    graph_order = len(theta_tensor)
    graph_nodes_and_all_possible_edges = len(layer2_biases)

    ##################################################################3################################################
    ################################3 initialize NN ##################################################################3
    ##################################################################3################################################

    NN = torch.nn.Sequential()

    ###############################
    ## Theta Layer initialization
    ###############################
    theta_layer = ElementwiseMultiply(in_features=graph_order, out_features=graph_order)

    # Temporarily disable gradient calc to set initial weights
    with torch.no_grad():
        theta_layer.weight.data = theta_tensor

    NN.append(theta_layer)

    ################################
    ## Second Layer initialization
    ################################
    layer2 = torch.nn.Linear(
        in_features=graph_order,
        out_features=graph_nodes_and_all_possible_edges,
        bias=True,
        dtype=torch_dtype
    )
    # add ReLu activation layer to layer 2
    layer2_activation = torch.nn.ReLU()

    # make layer non-trainable
    layer2.requires_grad_(False)

    # Initialize weights and biases
    layer2.weight.data = numpy.transpose(layer2_weights)
    layer2.bias.data = layer2_biases

    NN.append(layer2)
    NN.append(layer2_activation)

    ###############################
    ## Third Layer initialization
    ###############################
    layer3 = torch.nn.Linear(
        in_features=graph_nodes_and_all_possible_edges, out_features=1, bias=False
    )

    # make layer non-trainable
    layer3.requires_grad_(False)

    # Initialize weights
    layer3.weight.data = layer3_weights
    
    NN.append(layer3)

    return NN