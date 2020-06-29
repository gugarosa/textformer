import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """An Attention class is used to provide attention-based mechanisms in a neural network layer.

    References:
        D. Bahdanau, K. Cho, Y. Bengio. Neural machine translation by jointly learning to align and translate.
        Preprint arXiv:1409.0473 (2014).

    """

    def __init__(self, n_hidden_enc, n_hidden_dec):
        """Initialization method.

        Args:
            n_hidden_enc (int): Number of hidden units in the Encoder.
            n_hidden_dec (int): Number of hidden units in the Decoder.

        """

        # Overriding its parent class
        super(Attention, self).__init__()

        # Defining the energy-based layer
        self.e = nn.Linear(n_hidden_enc * 2 + n_hidden_dec, n_hidden_dec)

        # Defining the weight-based layer
        self.v = nn.Linear(n_hidden_dec, 1, bias=False)

    def forward(self, o, h):
        """Performs a forward pass over the layer.

        Args:
            o (torch.Tensor): Tensor containing the encoded outputs.
            h (torch.Tensor): Tensor containing the hidden states.

        Returns:
            The attention-based weights.

        """

        # Repeating the decoder hidden states as its smaller than the encoder ones
        hidden = h.unsqueeze(1).repeat(1, o.shape[0], 1)

        # Permuting the outputs
        encoder_outputs = o.permute(1, 0, 2)

        # Calculating the energy between decoder hidden state and encoder hidden states
        energy = torch.tanh(self.e(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculating the attention
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
