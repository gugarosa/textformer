import torch
from torch import distributions

import textformer.utils.logging as l
from textformer.core.model import Model
from textformer.models.decoders import AttentionBiGRUDecoder
from textformer.models.encoders import BiGRUEncoder

logger = l.get_logger(__name__)


class AttentionSeq2Seq(Model):
    """An AttentionSeq2Seq class implements an attention-based Sequence-To-Sequence learning architecture.

    """

    def __init__(self, n_input=128, n_output=128, n_hidden_enc=128, n_hidden_dec=128, n_embedding=128, dropout=0.5,
                 ignore_token=None, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_output (int): Number of output units.
            n_hidden_enc (int): Number of hidden units in the Encoder.
            n_hidden_dec (int): Number of hidden units in the Decoder.
            n_embedding (int): Number of embedding units.
            dropout (float): Amount of dropout to be applied.
            ignore_token (int): The index of a token to be ignore by the loss function.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        logger.info('Overriding class: Model -> AttentionSeq2Seq.')

        # Creating the encoder network
        E = BiGRUEncoder(n_input, n_hidden_enc,
                         n_hidden_dec, n_embedding, dropout)

        # Creating the decoder network
        D = AttentionBiGRUDecoder(
            n_output, n_hidden_enc, n_hidden_dec, n_embedding, dropout)

        # Overrides its parent class with any custom arguments if needed
        super(AttentionSeq2Seq, self).__init__(
            E, D, ignore_token, init_weights, device)

        logger.info('Class overrided.')

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.
            y (torch.Tensor): Tensor containing the true labels.
            teacher_forcing_ratio (float): Whether the next prediction should come
                from the predicted sample or from the true labels.

        Returns:
            The predictions over the input tensor.

        """

        # Creates an empty tensor to hold the predictions
        preds = torch.zeros(y.shape[0], y.shape[1], self.D.n_output, device=self.device)

        # Performs the initial encoding
        outputs, hidden = self.E(x)

        # Make sure that the first decoding will come from the true labels
        x = y[0, :]

        # For every possible token in the sequence
        for t in range(1, y.shape[0]):
            # Decodes the tensor
            pred, hidden = self.D(x, hidden, outputs)

            # Gathers the prediction of current token
            preds[t] = pred

            # Calculates whether teacher forcing should be used or not
            teacher_forcing = torch.rand(1,) < teacher_forcing_ratio

            # If teacher forcing should be used
            if teacher_forcing:
                # Gathers the new input from the true labels
                x = y[t]

            # If teacher forcing should not be used
            else:
                # Gathers the new input from the best prediction
                x = pred.argmax(1)

        return preds

    def sample(self, field, start, length=10, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            field (torchtext.data.Field): Datatype instructions for tensor convertion.
            start (str): The start string to generate the text.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list of generated text.

        """

        logger.debug(f'Generating text with length: {length} ...')

        # Setting the evalution flag
        self.eval()

        # Pre-processing the start text into tokens
        tokens = field.preprocess(start)

        # Numericalizing the tokens
        tokens = field.numericalize([tokens])

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # Performs the initial encoding
            outputs, hidden = self.E(tokens)

        # Removes the batch dimension from the tokens
        tokens = tokens.squeeze(0)

        # For every possible length
        for i in range(length):
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, hidden = self.D(tokens[-1], hidden, outputs)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a token from a categorical distribution based on the predictions
            sampled_token = distributions.Categorical(logits=preds).sample()

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token.unsqueeze(0)))

        # Decodes the tokens into text
        sampled_text = [field.vocab.itos[t] for t in tokens]

        return sampled_text
