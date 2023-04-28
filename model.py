from torch import nn
import torch

class FullyConnectedModel(nn.Module):
    """
    linear model, fully connected NN, allows for non-linearities via ReLU
    -> built by stacking layer blocks with a for looop, each block consists of a linear layer followed by a non-linear activation function"""
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_per_layer, dropout_rate=0.1):
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

class HalfingModel(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=2, dropout_rate=0.1):
        super(HalfingModel, self).__init__()

        layers = []

        in_features = input_size * 64

        # Initial layer
        layers.append(nn.Linear(input_size, in_features))
        layers.append(nn.ReLU())



        for _ in range(num_blocks):
            # Half the number of features after every block
            block = nn.Sequential(
                nn.Linear(in_features, in_features//2),
                nn.BatchNorm1d(in_features//2),
                nn.ReLU(),


                nn.Linear( in_features//2,  in_features//4),
                nn.BatchNorm1d(in_features//4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            layers.append(block)

            in_features = in_features // 4
            print(f"block: {block}, in_features: {in_features}")

        self.blocks = nn.Sequential(*layers)

        # Final block
        # @GPT: do it in the same style as above
        final_block = nn.Sequential(
            nn.Linear(in_features, input_size*2),
            nn.ReLU(),

            nn.Linear(input_size*2, input_size//2),
            nn.ReLU(),

            nn.Linear(input_size//2, output_size)
        )
        self.final_block = final_block

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_block(x)
        return x

#####################
# LSTM 1
#####################
class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#####################
# LSTM 2
#####################
# code taken from: https://www.kaggle.com/code/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch

# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
