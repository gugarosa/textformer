import math

import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Encoder

logger = l.get_logger(__name__)


class ConvEncoder(Encoder):
    """A ConvEncoder is used to supply the encoding part of the Convolutional Seq2Seq architecture.

    """

    def __init__(self, n_input=128, n_hidden=128, n_embedding=128, n_layers=1,
                 kernel_size=3, dropout=0.5, scale=0.5, max_length=100):
        """Initializion method.

        Args:
            n_input (int): Number of input units.
            n_hidden (int): Number of hidden units.
            n_embedding (int): Number of embedding units.
            n_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernels.
            dropout (float): Amount of dropout to be applied.
            scale (float): Value for the residual learning.
            max_length (int): Maximum length of positional embeddings.

        """

        logger.info('Overriding class: Encoder -> ConvEncoder.')

        # Overriding its parent class
        super(ConvEncoder, self).__init__()

        # Number of input units
        self.n_input = n_input

        # Number of hidden units
        self.n_hidden = n_hidden

        # Number of embedding units
        self.n_embedding = n_embedding

        # Number of layers
        self.n_layers = n_layers

        # Checks if kernel size is even
        if kernel_size % 2 == 0:
            # If yes, adds one to make it odd
            self.kernel_size = kernel_size + 1
        
        # If it is odd
        else:
            # Uses the inputted kernel size
            self.kernel_size = kernel_size

        # Maximum length of positional embeddings
        self.max_length = max_length

        # Scale for the residual learning
        self.scale = math.sqrt(scale)

        # Embedding layers
        self.embedding = nn.Embedding(n_input, n_embedding)
        self.pos_embedding = nn.Embedding(max_length, n_embedding)

        # Fully connected layers
        self.fc1 = nn.Linear(n_embedding, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embedding)

        # Convolutional layers
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=n_hidden,
                                             out_channels=2 * n_hidden,
                                             kernel_size=self.kernel_size,
                                             padding=(self.kernel_size - 1) // 2)
                                   for _ in range(n_layers)])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(f'Size: ({self.n_input}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.conv}.')

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The convolutions and output values.

        """

        # Creates the positions tensor
        pos = torch.arange(0, x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1)

        # Calculates the embedded outputs
        x_embedded = self.embedding(x)
        pos_embedded = self.pos_embedding(pos)

        # Combines the embeddings
        embedded = self.dropout(x_embedded + pos_embedded)

        # Passing down to the first linear layer and permuting its dimension
        hidden = self.fc1(embedded).permute(0, 2, 1)

        # For every convolutional layer
        for c in self.conv:
            # Pass down through convolutional layer
            conv = c(self.dropout(hidden))

            # Activates with a GLU function
            conv = nn.functional.glu(conv, dim=1)

            # Sums the activation with its residual learning
            conv = (conv + hidden) * self.scale

            # Puts back to the next layer input
            hidden = conv

        # Passes down back to embedding size
        conv = self.fc2(conv.permute(0, 2, 1))

        # Sums the embedded features with the convolutional-extracted ones
        output = (conv + embedded) * self.scale

        return conv, output
