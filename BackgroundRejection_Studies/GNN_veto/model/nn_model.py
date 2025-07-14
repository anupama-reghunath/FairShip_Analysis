import torch
import torch.nn as nn

from torch import nn


class NN(nn.Module):
    """
    Basic class for a customisable NN model in pytorch
    """
    def __init__(self, input_dim = 2003, output_dim = 3, hidden_sizes = [128, 64, 32, 16, 8], dropout = 0.0):
        """
        Constructor for NN class.

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_sizes (list): list of number of hidden units (int) of the hidden layers
            dropout (float): dropout probability
        """
        super(NN, self).__init__()
        self._layers = []
        self._batch_norms = []

        for i in range(len(hidden_sizes) -1):
            if i == 0:
                self._layers.append(nn.Linear(input_dim, hidden_sizes[i+1]))
            else:
                self._layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

            self._batch_norms.append(nn.BatchNorm1d(hidden_sizes[i+1]))

        self._layers = nn.ModuleList(self._layers)
        self._batch_norms = nn.ModuleList(self._batch_norms)
        self._dropout = nn.Dropout(dropout)
        self._output_layer = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        """
        Forward pass function of the network.

        Args:
            x (torch.tensor): input to the network
        """
        for i in range(len(self._layers)):
            x = self._batch_norms[i](torch.relu(self._layers[i](x)))
            x = self._dropout(x)
        x = self._output_layer(x)
        return x

    def load_model(self, path, device=None):
        
        """
        Load the model state from a file and move it to the appropriate device.
        
        Args:
            path (str): Path to the model file.
            device (torch.device, optional): Device to load the model onto. If None, it will be set based on CUDA availability.
            
        Returns:
            NN: The current model instance with loaded weights.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
        self.eval()
        return self