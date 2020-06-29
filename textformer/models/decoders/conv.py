import math

import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Decoder
from textformer.models.layers import ResidualAttention

logger = l.get_logger(__name__)


class ConvDecoder(Decoder):
    def __init__(self, n_output=128, n_hidden=128, n_embedding=128, n_layers=1,
                 kernel_size=3, dropout=0.5, scale=0.5, max_length=100, pad_token=None):
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
            pad_token (int): The index of a padding token.

        """

        logger.info('Overriding class: Encoder -> ConvDecoder.')

        # Overriding its parent class
        super(ConvDecoder, self).__init__()

        # Number of output units
        self.n_output = n_output

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

        # Padding token index
        self.pad_token = pad_token

        # Embedding layers
        self.embedding = nn.Embedding(n_output, n_embedding)
        self.pos_embedding = nn.Embedding(max_length, n_embedding)
        
        # Fully connected layers
        self.fc1 = nn.Linear(n_embedding, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embedding)
                
        # Residual Attention layer
        self.a = ResidualAttention(n_hidden, n_embedding, self.scale)
                
        # Convolutional layers
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=n_hidden, 
                                              out_channels=2 * n_hidden, 
                                              kernel_size=self.kernel_size)
                                    for _ in range(n_layers)])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.out = nn.Linear(n_embedding, n_output)

        logger.debug(f'Size: ({self.n_output}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.conv}.')
        
    def forward(self, y, c, o):
        """Performs a forward pass over the architecture.

        Args:
            y (torch.Tensor): Tensor containing the true labels.
            c (torch.Tensor): Tensor containing the convolutional features.
            o (torch.Tensor): Tensor containing combined outputs.

        Returns:
            The output and attention values.

        """

        # Creates the positions tensor
        pos = torch.arange(0, y.shape[1]).unsqueeze(0).repeat(y.shape[0], 1)

        # Calculates the embedded outputs
        y_embedded = self.embedding(y)
        pos_embedded = self.pos_embedding(pos)
        
        # Combines the embeddings
        embedded = self.dropout(y_embedded + pos_embedded)

        # Passing down to the first linear layer and permuting its dimension
        hidden = self.fc1(embedded).permute(0, 2, 1)
        
        # For every convolutional layer
        for layer in self.conv:
            # Applying dropout
            hidden = self.dropout(hidden)
        
            # Padding tensor
            pad = torch.zeros((hidden.shape[0], hidden.shape[1], self.kernel_size - 1))
        
            # If padding token exists
            if self.pad_token:
                # Fills with its index
                pad = pad.fill_(self.pad_token)
            
            # Concatenating padding and convolutional features
            conv = torch.cat((pad, hidden), dim=2)
        
            # Pass down through convolutional layer
            conv = layer(conv)

            # Activates with a GLU function
            conv = nn.functional.glu(conv, dim=1)

            # Calculating attention
            attention, conv = self.a(embedded, conv, c, o)
            
            # Applying residual connections
            conv = (conv + hidden) * self.scale
            
            # Puts back to the next layer input
            hidden = conv

        # Passes down back to embedding size
        conv = self.fc2(conv.permute(0, 2, 1))
            
        # Calculates the outputs
        output = self.out(self.dropout(conv))
            
        return output, attention
