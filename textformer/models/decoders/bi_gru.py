import torch
from torch import nn

import textformer.utils.logging as l
from textformer.core import Decoder
from textformer.models.layers import Attention

logger = l.get_logger(__name__)


class BiGRUDecoder(Decoder):
    """A BiGRUDecoder class is used to supply the decoding part of the Attention-based Seq2Seq architecture.

    """

    def __init__(self, n_output=128, n_hidden_enc=128, n_hidden_dec=128, n_embedding=128, dropout=0.5):
        """Initialization method.

        Args:
            n_output (int): Number of output units.
            n_hidden_enc (int): Number of hidden units in the Encoder.
            n_hidden_dec (int): Number of hidden units in the Decoder.
            n_embedding (int): Number of embedding units.
            dropout (float): Amount of dropout to be applied.

        """

        logger.info('Overriding class: Decoder -> BiGRUDecoder.')

        # Overriding its parent class
        super(BiGRUDecoder, self).__init__()

        # Number of output units
        self.n_output = n_output

        # Number of hidden units
        self.n_hidden = n_hidden_dec

        # Number of embedding units
        self.n_embedding = n_embedding

        # Embedding layer
        self.embedding = nn.Embedding(n_output, n_embedding)

        # Attention layer
        self.a = Attention(n_hidden_enc, n_hidden_dec)

        # RNN layer
        self.rnn = nn.GRU(n_hidden_enc * 2 + n_embedding, n_hidden_dec)

        # Fully connected layer
        self.fc = nn.Linear(n_hidden_enc * 2 + n_hidden_dec + n_embedding, n_output)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f'Size: ({self.n_output}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.rnn} | Attention: {self.a} | Output: {self.fc}.')

    def forward(self, x, o, h):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the input data.
            o (torch.Tensor): Tensor containing the encoded outputs.
            h (torch.Tensor): Tensor containing the hidden states.

        Returns:
            The prediction and hidden state.

        """

        # Calculates the embedded layer
        embedded = self.dropout(self.embedding(x.unsqueeze(0)))

        # Calculates the attention
        attention = self.a(o, h).unsqueeze(1)

        # Permutes the encoder outputs
        encoder_outputs = o.permute(1, 0, 2)

        # Calculates the weights from the attention-based layer
        weighted = torch.bmm(attention, encoder_outputs).permute(1, 0, 2)

        # Calculates the RNN layer
        output, hidden = self.rnn(torch.cat((embedded, weighted), dim=2), h.unsqueeze(0))

        # Concatenating the output with hidden and context tensors
        output = torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1)

        # Calculates the prediction over the fully connected layer
        pred = self.fc(output)

        return pred, hidden.squeeze(0), attention.squeeze(1)
