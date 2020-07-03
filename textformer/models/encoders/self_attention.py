import math

import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Encoder
from textformer.models.layers import MultiHeadAttention, PositionWideForward

logger = l.get_logger(__name__)


class SelfAttentionLayer(nn.Module):
    """A SelfAttentionLayer is used to supply a self-attention layer to the encoding part of the Transformer architecture.

    """

    def __init__(self, n_hidden=128, n_forward=256, n_heads=3, dropout=0.1):
        """Initialization method.

        Args:
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of feed forward units.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.

        """

        # Overriding its parent class
        super(SelfAttentionLayer, self).__init__()

        # Normalization layers
        self.norm1 = nn.LayerNorm(n_hidden)
        self.norm2 = nn.LayerNorm(n_hidden)

        # Multi-head attention layer
        self.att = MultiHeadAttention(n_hidden, n_heads, dropout)

        # Position-wide feed forward layer
        self.pw = PositionWideForward(n_hidden, n_forward, dropout)

        # Dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.
            x_mask (torch.Tensor): Tensor containing the masked data.

        Returns:
            The output values.

        """

        # Performs the self-attention mechanism
        _x, _ = self.att(x, x, x, x_mask)

        # Performs the dropout with residual connection and layer normalization
        x_norm = self.norm1(x + self.drop(_x))

        # Performs the position-wise forwarding
        pos_wide = self.pw(x_norm)

        # Performs the dropout with residual connection and layer normalization
        residual_attention = self.norm2(x_norm + self.drop(pos_wide))

        return residual_attention


class SelfAttentionEncoder(Encoder):
    """A SelfAttentionEncoder is used to supply the encoding part of the Transformer architecture.

    """

    def __init__(self, n_input=128, n_hidden=128, n_forward=256, n_layers=1,
                 n_heads=3, dropout=0.1, max_length=100):
        """Initializion method.

        Args:
            n_input (int): Number of input units.
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of feed forward units.
            n_layers (int): Number of attention layers.
            n_heads (int): Number of attention heads.
            dropout (float): Amount of dropout to be applied.
            max_length (int): Maximum length of positional embeddings.

        """

        logger.info('Overriding class: Encoder -> SelfAttentionEncoder.')

        # Overriding its parent class
        super(SelfAttentionEncoder, self).__init__()

        # Number of input units
        self.n_input = n_input

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

        # Encoding layers
        self.encoders = nn.ModuleList([SelfAttentionLayer(n_hidden, n_heads, n_forward, dropout) for _ in range(n_layers)])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.
            x_mask (torch.Tensor): Tensor containing the masked data.

        Returns:
            The output values.

        """

        # Creates the positions tensor
        pos = torch.arange(0, x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1)

        # Calculates the embedded outputs
        token_embedded = self.embedding(x)
        pos_embedded = self.pos_embedding(pos)

        # Combines the embeddings
        embedded = self.dropout(token_embedded * self.scale + pos_embedded)

        # For every self-attention layer
        for layer in self.encoders:
            # Pass down through layer
            embedded = layer(embedded, x_mask)

        return embedded
