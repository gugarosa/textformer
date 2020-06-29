import torch.nn as nn
import torch.nn.functional as F


class PositionWideForward(nn.Module):
    """A PositionWideForward class is used to provide a position-wise feed forward layer for a neural network.

    References:
        A. Vaswani, et al. Attention is all you need. Advances in neural information processing systems (2017).

    """

    def __init__(self, n_hidden, n_forward, dropout):
        """Initialization method.

        Args:
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of forward units.
            dropout (float): Dropout probability.

        """

        # Overriding its parent class
        super(PositionWideForward, self).__init__()

        # Defining the linear (feed forward) layers
        self.fc1 = nn.Linear(n_hidden, n_forward)
        self.fc2 = nn.Linear(n_forward, n_hidden)

        # Defining the dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Performs a forward pass over the layer.

        Args:
            x (torch.Tensor): Tensor containing the input states.

        Returns:
            The feed forward activations.

        """

        # Performs the pass over first linear layer and activates using ReLU
        x = F.relu(self.fc1(x))

        # Pass down to the dropout layer
        x = self.drop(x)

        # Pass down over the second linear layer
        x = self.fc2(x)

        return x
