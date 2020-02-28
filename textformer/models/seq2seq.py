import textformer.utils.logging as l
import torch
from torch import nn
import random
from textformer.core.model import Model

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

    def forward(self, x, h, c):
        """
        """

        #
        x = x.unsqueeze(0)

        #
        embedded = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.rnn(embedded, (h, c))

        pred = self.fc(output.squeeze(0))

        return pred, hidden, cell


class Seq2Seq(Model):
    """
    """

    def __init__(self, encoder, decoder):
        """
        """

        # Overriding its parent class
        super(Seq2Seq, self).__init__()

        #
        self.encoder = encoder

        #
        self.decoder = decoder

        #
        self.optimizer = torch.optim.Adam(self.parameters())

        #
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
        """

        #
        batch_size = y.shape[1]

        #
        target_size = y.shape[0]

        #
        target_vocab_size = self.decoder.n_output

        #
        outputs = torch.zeros(target_size, batch_size, target_vocab_size)

        #
        hidden, cell = self.encoder(x)

        #
        x = y[0, :]

        #
        for t in range(1, target_size):
            #
            output, hidden, cell = self.decoder(x, hidden, cell)

            #
            outputs[t] = output

            #
            teacher_forcing = random.random() < teacher_forcing_ratio

            #
            top_pred = output.argmax(1)

            #
            if teacher_forcing:
                x = y[t]
            else:
                x = top_pred

        return outputs
