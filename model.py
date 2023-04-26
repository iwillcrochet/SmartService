from torch import nn


class FullyConnectedModel(nn.Module):
    """
    linear model, fully connected NN, allows for non-linearities via ReLU
    -> built by stacking layer blocks with a for looop, each block consists of a linear layer followed by a non-linear activation function"""
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_per_layer):
        super(FullyConnectedModel, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, nodes_per_layer))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(nodes_per_layer, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
