import math

import textformer.utils.logging as l
import torch
from textformer.core import Encoder
from textformer.models.layers import MultiHeadAttention, PositionWideForward
from torch import nn

logger = l.get_logger(__name__)


class SelfAttentionLayer(nn.Module):
    """A SelfAttentionLayer is used to supply the self-attention layer to the encoding part of the Transformer architecture.

    """

    def __init__(self, n_hidden=128, n_forward=256, n_heads=3, dropout=0.1):
        """Initialization method.

        Args:
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of feed forward units.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.

        """

        #
        self.self_attn_layer_norm = nn.LayerNorm(n_hidden)

        #
        self.ff_layer_norm = nn.LayerNorm(n_hidden)

        #
        self.self_attention = MultiHeadAttention(n_hidden, n_heads, dropout)

        #
        self.positionwise_feedforward = PositionWideForward(
            n_hidden, n_forward, dropout)

        #
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        """

        # Performs the self-attention mechanism
        _src, _ = self.self_attention(src, src, src, src_mask)

        # Performs the dropout with residual connection and layer normalization
        src = self.self_attn_layer_norm(src + self.drop(_src))

        # Performs the position-wise forwarding
        _src = self.positionwise_feedforward(src)

        # Performs the dropout with residual connection and layer normalization
        src = self.ff_layer_norm(src + self.drop(_src))

        return src


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
        self.encoders = nn.ModuleList[SelfAttentionLayer(n_hidden, n_heads, n_forward, dropout) for _ in range(n_layers)]

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

        pass
