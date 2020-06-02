import torch
from torch import distributions
from torchtext.data.metrics import bleu_score

import textformer.utils.logging as l
from textformer.core.model import Model
from textformer.models.decoders import ConvDecoder
from textformer.models.encoders import ConvEncoder

logger = l.get_logger(__name__)


class ConvSeq2Seq(Model):
    """A ConvSeq2Seq class implements a Convolutional Sequence-To-Sequence learning architecture.

    References:
        J. Gehring, et al. Convolutional sequence to sequence learning.
        Proceedings of the 34th International Conference on Machine Learning (2017).

    """

    def __init__(self, n_input=128, n_output=128, n_hidden=128, n_embedding=128, n_layers=1, kernel_size=3, 
                 dropout=0.5, scale=0.5, max_length=100, ignore_token=None, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_output (int): Number of output units.
            n_hidden (int): Number of hidden units.
            n_embedding (int): Number of embedding units.
            n_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernels.
            dropout (float): Amount of dropout to be applied.
            scale (float): Value for the residual learning.
            max_length (int): Maximum length of positional embeddings.
            ignore_token (int): The index of a token to be ignored by the loss function.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        logger.info('Overriding class: Model -> ConvSeq2Seq.')

        # Creating the encoder network
        E = ConvEncoder(n_input, n_hidden, n_embedding, n_layers, kernel_size, dropout, scale, max_length)

        # Creating the decoder network
        D = ConvDecoder(n_output, n_hidden, n_embedding, n_layers, kernel_size, dropout, scale, max_length, ignore_token)

        # Overrides its parent class with any custom arguments if needed
        super(ConvSeq2Seq, self).__init__(E, D, ignore_token, init_weights, device)

        logger.info('Class overrided.')

    def forward(self, x, y, teacher_forcing_ratio=0.0):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.
            y (torch.Tensor): Tensor containing the true labels.
            teacher_forcing_ratio (float): Whether the next prediction should come
                from the predicted sample or from the true labels.

        Returns:
            The predictions over the input tensor.

        """

        # Performs the encoding
        conv, output = self.E(x)

        # Decodes the encoded inputs
        preds, _ = self.D(y, conv, output)

        return preds
