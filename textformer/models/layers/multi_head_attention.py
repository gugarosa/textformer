import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import textformer.utils.constants as c


class MultiHeadAttention(nn.Module):
    """A MultiHeadAttention class is used to provide multi-head attention-based mechanisms in a neural network layer.

    References:
        A. Vaswani, et al. Attention is all you need. Advances in neural information processing systems (2017).

    """

    def __init__(self, n_hidden, n_heads, dropout):
        """Initialization method.

        Args:
            n_hidden (int): Number of hidden units.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.

        """

        # Overriding its parent class
        super(MultiHeadAttention, self).__init__()

        # Asserts if number of hidden units is divisible by number of heads
        assert n_hidden % n_heads == 0

        # Number of hidden units
        self.n_hidden = n_hidden

        # Number of attention heads
        self.n_heads = n_heads

        # Size of attention head
        self.head_size = n_hidden // n_heads

        # Linear projections (query, key and value)
        self.q = nn.Linear(n_hidden, n_hidden)
        self.k = nn.Linear(n_hidden, n_hidden)
        self.v = nn.Linear(n_hidden, n_hidden)

        # Output projection
        self.out = nn.Linear(n_hidden, n_hidden)

        # Dropout layer
        self.drop = nn.Dropout(dropout)

        # Scale for the residual connections
        self.scale = math.sqrt(self.head_size)

    def forward(self, query, key, value, mask=None):
        """Performs a forward pass over the layer.

        Args:
            q (torch.Tensor): Tensor containing the queries.
            k (torch.Tensor): Tensor containing the keys.
            v (torch.Tensor): Tensor containing the values.
            m (torch.Tensor): Tensor containing the mask.

        Returns:
            The multi-head attention-based weights.

        """

        # Gathers the batch size
        batch_size = query.shape[0]

        # Performs the linear projections to calculate Q, K and V
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)

        # Reshapes Q, K and V
        Q = Q.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        # Calculates the energy
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Checks if a mask is supplied
        if mask is not None:
            # Fills the energy with a low value  where mask equals to zero
            energy = energy.masked_fill(mask == 0, -c.EPSILON)

        # Calculates the attention
        attention = torch.softmax(energy, dim=-1)

        # Performs the energy-value projection
        x = (torch.matmul(self.drop(attention), V)).permute(0, 2, 1, 3)

        # Reshapes back to hidden units
        x = x.view(batch_size, -1, self.n_hidden)

        # Passes down through output layer
        x = self.out(x)

        return x, attention
