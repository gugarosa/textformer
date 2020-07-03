import math

import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Decoder
from textformer.models.layers import MultiHeadAttention, PositionWideForward

logger = l.get_logger(__name__)

class SelfAttentionLayer(nn.Module):
    """A SelfAttentionLayer is used to supply a self-attention layer to the decoding part of the Transformer architecture.

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
        self.norm3 = nn.LayerNorm(n_hidden)

        # Multi-head attention layers
        self.self_att = MultiHeadAttention(n_hidden, n_heads, dropout)
        self.enc_att = MultiHeadAttention(n_hidden, n_heads, dropout)

        # Position-wide feed forward layer
        self.pw = PositionWideForward(n_hidden, n_forward, dropout)

        # Dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, y, y_mask, x, x_mask):
        """Performs a forward pass over the architecture.

        Args:
            y (torch.Tensor): Tensor containing the true labels.
            y_mask (torch.Tensor): Tensor containing the masked labels.
            x_enc (torch.Tensor): Tensor containing the encoded data.
            x_mask (torch.Tensor): Tensor containing the masked data.

        Returns:
            The output values.

        """

        # Performs the self-attention mechanism
        _y, _ = self.self_att(y, y, y, y_mask)

        # Performs the dropout with residual connection and layer normalization
        y_norm = self.norm1(y + self.drop(_y))

        # Performs the self-attention mechanism (encoder)
        _y, attention = self.enc_att(y_norm, x, x, x_mask)

        # Performs the dropout with residual connection and layer normalization (encoder)
        y_norm = self.norm2(y_norm + self.drop(_y))

        # Performs the position-wise forwarding
        pos_wide = self.pw(y_norm)

        # Performs the dropout with residual connection and layer normalization
        y_norm = self.norm3(y_norm + self.drop(pos_wide))

        return y_norm, attention


class SelfAttentionDecoder(Decoder):
    """A SelfAttentionDecoder is used to supply the decoding part of the Transformer architecture.

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

        logger.info('Overriding class: Encoder -> SelfAttentionDecoder.')

        # Overriding its parent class
        super(SelfAttentionDecoder, self).__init__()

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
        self.embedding = nn.Embedding(n_output, n_hidden)
        self.pos_embedding = nn.Embedding(max_length, n_hidden)

        # Decoding layers
        self.decoders = nn.ModuleList([SelfAttentionLayer(n_hidden, n_heads, n_forward, dropout) for _ in range(n_layers)])
        
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

        # Creates the positions tensor
        pos = torch.arange(0, y.shape[1]).unsqueeze(0).repeat(y.shape[0], 1)

        # Calculates the embedded outputs
        token_embedded = self.embedding(y)
        pos_embedded = self.pos_embedding(pos)

        # Combines the embeddings
        embedded = self.dropout(token_embedded * self.scale + pos_embedded)

        # For every self-attention layer
        for layer in self.decoders:
            # Pass down through layer
            embedded, attention = layer(embedded, y_mask, x, x_mask)

        # Pass through output layer
        output = self.out(embedded)

        return output, attention
