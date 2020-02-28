import textformer.utils.logging as l
from torch import nn

logger = l.get_logger(__name__)

class Decoder(nn.Module):
    """
    """

    def __init__(self, n_output=128, n_hidden=128, n_embedding=128, n_layers=1, dropout=0.5):
        """
        """

        # Overriding its parent class
        super(Decoder, self).__init__()

        #
        self.n_output = n_output

        #
        self.embedding = nn.Embedding(n_output, n_embedding)

        #
        self.rnn = nn.LSTM(n_embedding, n_hidden, n_layers, dropout=dropout)

        #
        self.fc = nn.Linear(n_hidden, n_output)

        #
        self.dropout = nn.Dropout(dropout)

    def foward(self, x, h, c):
        """
        """

        #
        x = x.unsqueeze(0)

        #
        embedded = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.rnn(embedded, (h, c))

        pred = self.fc(output.squeeze(0))

        return pred, hidden, cell
