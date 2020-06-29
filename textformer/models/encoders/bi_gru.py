import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Encoder

logger = l.get_logger(__name__)


class BiGRUEncoder(Encoder):
    """A BiGRUEncoder is used to supply the encoding part of the Attention-based Seq2Seq architecture.

    """

    def __init__(self, n_input=128, n_hidden_enc=128, n_hidden_dec=128, n_embedding=128, dropout=0.5):
        """Initializion method.

        Args:
            n_input (int): Number of input units.
            n_hidden_enc (int): Number of hidden units in the Encoder.
            n_hidden_dec (int): Number of hidden units in the Decoder.
            n_embedding (int): Number of embedding units.
            dropout (float): Amount of dropout to be applied.

        """

        logger.info('Overriding class: Encoder -> BiGRUEncoder.')

        # Overriding its parent class
        super(BiGRUEncoder, self).__init__()

        # Number of input units
        self.n_input = n_input

        # Number of hidden units
        self.n_hidden = n_hidden_enc

        # Number of embedding units
        self.n_embedding = n_embedding

        # Embedding layer
        self.embedding = nn.Embedding(n_input, n_embedding)

        # RNN layer
        self.rnn = nn.GRU(n_embedding, n_hidden_enc, bidirectional=True)

        # Fully-connected layer
        self.fc = nn.Linear(n_hidden_enc * 2, n_hidden_dec)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f'Size: ({self.n_input}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.rnn} | Output: {self.fc}.')

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The hidden state and cell values.

        """

        # Calculates the embedded layer outputs
        embedded = self.dropout(self.embedding(x))

        # Calculates the RNN outputs
        outputs, hidden = self.rnn(embedded)

        # Calculates the final hidden state of the encoder forward and backward RNNs
        # Also, they are fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden
