import textformer.utils.logging as l
import torch
import random
from textformer.core.model import Model

logger = l.get_logger(__name__)

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
