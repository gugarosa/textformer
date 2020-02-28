import textformer.utils.logging as l
from torch import nn

logger = l.get_logger(__name__)

class Encoder(nn.Module):
    """
    """

    def __init__(self, n_input=128, n_hidden=128, n_embedding=128, n_layers=1, dropout=0.5):
        """
        """

        # Overriding its parent class
        super(Encoder, self).__init__()

        #
        # self.n_input = n_input

        # #
        # self.n_hidden = n_hidden

        # #
        # self.n_embedding = n_embedding

        # #
        # self.n_layers = n_layers

        #
        self.embedding = nn.Embedding(n_input, n_embedding)

        #
        self.rnn = nn.LSTM(n_embedding, n_hidden, n_layers, dropout=dropout)

        #
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        """

        #
        embedded = self.dropout(self.embedding(x))

        #
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell
