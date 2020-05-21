from torch import nn

import textformer.utils.logging as l
from textformer.core import Encoder

logger = l.get_logger(__name__)


class LSTMEncoder(Encoder):
    """A LSTMEncoder is used to supply the encoding part of the Seq2Seq architecture.

    """

    def __init__(self, n_input=128, n_hidden=128, n_embedding=128, n_layers=1, dropout=0.5):
        """Initializion method.

        Args:
            n_input (int): Number of input units.
            n_hidden (int): Number of hidden units.
            n_embedding (int): Number of embedding units.
            n_layers (int): Number of RNN layers.
            dropout (float): Amount of dropout to be applied.

        """

        logger.info('Overriding class: Encoder -> LSTMEncoder.')

        # Overriding its parent class
        super(LSTMEncoder, self).__init__()

        # Number of input units
        self.n_input = n_input

        # Number of hidden units
        self.n_hidden = n_hidden

        # Number of embedding units
        self.n_embedding = n_embedding

        # Number of layers
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(n_input, n_embedding)

        # RNN layer
        self.rnn = nn.LSTM(n_embedding, n_hidden, n_layers, dropout=dropout)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f'Size: ({self.n_input}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.rnn}.')

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
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell
