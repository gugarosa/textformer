import torch
from torch import distributions
from torchtext.data.metrics import bleu_score

import textformer.utils.logging as l
from textformer.core.model import Model
from textformer.models.decoders import SelfAttentionDecoder
from textformer.models.encoders import SelfAttentionEncoder

logger = l.get_logger(__name__)


class Transformer(Model):
    """A Transformer class implements a Transformer-based learning architecture.

    References:
        A. Vaswani, et al. Attention is all you need. Advances in neural information processing systems (2017).

    """

    def __init__(self, n_input=128, n_output=128, n_hidden=128, n_forward=256, n_layers=1, n_heads=3, 
                 dropout=0.1, max_length=100, source_pad_index=None, target_pad_index=None,
                 init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_output (int): Number of output units.
            n_hidden (int): Number of hidden units.
            n_forward (int): Number of feed forward units.
            n_layers (int): Number of attention layers.
            n_heads (int): Number of attention heads.
            dropout (float): Amount of dropout to be applied.
            max_length (int): Maximum length of positional embeddings.
            source_pad_index (int): The index of source vocabulary padding token.
            target_pad_index (int): The index of target vocabulary padding token.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        logger.info('Overriding class: Model -> Transformer.')

        # Creating the encoder network
        E = SelfAttentionEncoder(n_input, n_hidden, n_forward, n_layers, n_heads, dropout, max_length)

        # Creating the decoder network
        D = SelfAttentionDecoder(n_output, n_hidden, n_forward, n_layers, n_heads, dropout, max_length)

        # Overrides its parent class with any custom arguments if needed
        super(Transformer, self).__init__(E, D, None, init_weights, device)

        # Source vocabulary padding token
        self.source_pad_index = source_pad_index

        # Target vocabulary padding token
        self.target_pad_index = target_pad_index

        logger.info('Class overrided.')

    def create_source_mask(self, x):
        """
        """
        
        #
        x_mask = (x != self.source_pad_index).unsqueeze(1).unsqueeze(2)

        return x_mask

    def create_target_mask(self, y):
        """
        """
        
        #
        y_pad_mask = (y != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        
        #
        y_sub_mask = torch.tril(torch.ones((y.shape[1], y.shape[1]))).bool()
        
        #            
        y_mask = y_pad_mask & y_sub_mask
        
        return y_mask

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

        # Creates the source mask
        x_mask = self.create_source_mask(x)

        # Creates the target mask
        y_mask = self.create_target_mask(y)

        # Performs the encoding
        output = self.E(x, x_mask)

        # Decodes the encoded inputs
        preds, _ = self.D(y, y_mask, output, x_mask)

        return preds
