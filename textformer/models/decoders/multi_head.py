import math

import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Decoder

logger = l.get_logger(__name__)


class MultiHeadDecoder(Decoder):
    """A MultiHeadDecoder is used to supply the decoding part of the Transformer architecture.

    """

    def __init__(self, n_output=128, n_hidden=128, n_forward=256, n_layers=1,
                 n_heads=3, dropout=0.1, max_length=100):
        """Initializion method.

        Args:
            n_output (int): Number of output units.
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of feed forward units.
            n_layers (int): Number of attention layers.
            n_heads (int): Number of attention heads.
            dropout (float): Amount of dropout to be applied.
            max_length (int): Maximum length of positional embeddings.

        """

        logger.info('Overriding class: Encoder -> MultiHeadDecoder.')

        # Overriding its parent class
        super(MultiHeadDecoder, self).__init__()

        # Number of output units
        self.n_output = n_output

        # Number of hidden units
        self.n_hidden = n_hidden

        # Number of feed forward units
        self.n_forward = n_forward

        # Number of attention layers
        self.n_layers = n_layers

        # Number of attention heads
        self.n_heads = n_heads

        # Maximum length of positional embeddings
        self.max_length = max_length

        # Scale for the residual learning
        self.scale = math.sqrt(n_hidden)

        # Embedding layers
        self.embedding = nn.Embedding(n_input, n_hidden)
        self.pos_embedding = nn.Embedding(max_length, n_hidden)

        # Decoding layers

        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.out = nn.Linear(n_hidden, n_output)
        
    def forward(self, y, y_mask, x, x_mask):
        """Performs a forward pass over the architecture.

        Args:
            y (torch.Tensor): Tensor containing the true labels.
            y_mask (torch.Tensor): Tensor containing the masked labels.
            x_enc (torch.Tensor): Tensor containing the encoded data.
            x_mask (torch.Tensor): Tensor containing the masked data.

        Returns:
            The output and attention values.

        """

        pass