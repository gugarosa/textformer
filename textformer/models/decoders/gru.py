import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Decoder

logger = l.get_logger(__name__)


class GRUDecoder(Decoder):
    """A GRUDecoder class is used to supply the decoding part of the JointSeq2Seq architecture.

    """

    def __init__(self, n_output=128, n_hidden=128, n_embedding=128, dropout=0.5):
        """Initialization method.

        Args:
            n_output (int): Number of output units.
            n_hidden (int): Number of hidden units.
            n_embedding (int): Number of embedding units.
            dropout (float): Amount of dropout to be applied.

        """

        logger.info('Overriding class: Decoder -> GRUDecoder.')

        # Overriding its parent class
        super(GRUDecoder, self).__init__()

        # Number of output units
        self.n_output = n_output

        # Number of hidden units
        self.n_hidden = n_hidden

        # Number of embedding units
        self.n_embedding = n_embedding

        # Embedding layer
        self.embedding = nn.Embedding(n_output, n_embedding)

        # RNN layer
        self.rnn = nn.GRU(n_embedding + n_hidden, n_hidden)

        # Fully connected layer
        self.fc = nn.Linear(n_embedding + n_hidden * 2, n_output)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f'Size: ({self.n_output}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.rnn} | Output: {self.fc}.')

    def forward(self, x, h, c):
        """Performs a forward pass over the architecture.

        Args:
            x_enc (torch.Tensor): Tensor containing the input data.
            h (torch.Tensor): Tensor containing the hidden states.
            c (torch.Tensor): Tensor containing the context.

        Returns:
            The prediction and hidden state values.

        """

        # Calculates the embedded layer
        embedded = self.dropout(self.embedding(x.unsqueeze(0)))

        # Concatenating the embedding and context tensors
        concat_embedded = torch.cat((embedded, c), dim=2)

        # Calculates the RNN layer
        output, hidden = self.rnn(concat_embedded, h)

        # Concatenating the output with hidden and context tensors
        output = torch.cat((embedded, h, c), dim=2)

        # Calculates the prediction over the fully connected layer
        pred = self.fc(output.squeeze(0))

        return pred, hidden
