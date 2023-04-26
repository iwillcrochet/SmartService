from torch import nn

class FullyConnectedModel(nn.Module):
    """
    linear model, fully connected NN, allows for non-linearities via ReLU
    -> built by stacking layer blocks with a for looop, each block consists of a linear layer followed by a non-linear activation function"""
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_per_layer, dropout_rate=0.5):
        super(FullyConnectedModel, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, nodes_per_layer))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            self.layers.append(nn.ReLU())

            # Add a dropout layer after every 2 blocks except for the final block
            if (i + 1) % 2 == 0 and i != num_hidden_layers - 2:
                self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(nodes_per_layer, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x