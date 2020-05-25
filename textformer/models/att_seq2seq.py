import torch
from torch import distributions

import textformer.utils.logging as l
from textformer.core.model import Model
from textformer.models.decoders import AttBiGRUDecoder
from textformer.models.encoders import BiGRUEncoder

logger = l.get_logger(__name__)


class AttSeq2Seq(Model):
    """An AttSeq2Seq class implements an attention-based Sequence-To-Sequence learning architecture.

    References:
        D. Bahdanau, K. Cho, Y. Bengio. Neural machine translation by jointly learning to align and translate.
        Preprint arXiv:1409.0473 (2014).

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

        logger.info('Overriding class: Model -> AttSeq2Seq.')

        # Creating the encoder network
        E = BiGRUEncoder(n_input, n_hidden_enc, n_hidden_dec, n_embedding, dropout)

        # Creating the decoder network
        D = AttBiGRUDecoder(n_output, n_hidden_enc, n_hidden_dec, n_embedding, dropout)

        # Overrides its parent class with any custom arguments if needed
        super(AttSeq2Seq, self).__init__(E, D, ignore_token, init_weights, device)

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
            pred, hidden, _ = self.D(x, hidden, outputs)

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

    def generate_text(self, start, field, length=10, temperature=1.0):
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
        tokens = field.numericalize([tokens]).to(self.device)

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
                preds, hidden, _ = self.D(tokens[-1], hidden, outputs)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a token from a categorical distribution based on the predictions
            sampled_token = distributions.Categorical(logits=preds).sample()

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token.unsqueeze(0)))

        # Decodes the tokens into text
        sampled_text = [field.vocab.itos[t] for t in tokens]

        return sampled_text

    def translate_text(self, start, src_field, trg_field, max_length=10):
        """Translates text from the source vocabulary to the target vocabulary.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own translation implementation.

        Args:
            start (str): The string to be translated.
            src_field (torchtext.data.Field): Source vocabulary datatype instructions for tensor convertion.
            trg_field (torchtext.data.Field): Target vocabulary datatype instructions for tensor convertion.
            max_length (int): Maximum length of translated text.

        Returns:
            A list of translated text.

        """

        logger.debug(f'Translating text with maximum length: {max_length} ...')

        # Setting the evalution flag
        self.eval()

        # Pre-processing the start text into tokens
        tokens = src_field.preprocess(start)

        # Adding `<sos>`` and `<eos>` to the tokens
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        #
        atts = torch.zeros(max_length, 1, len(tokens)).to(self.device)

        # Numericalizing the tokens
        tokens = src_field.numericalize([tokens]).to(self.device)

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # Performs the initial encoding
            outputs, hidden = self.E(tokens)

        # Creating a tensor with `<sos>` token from target vocabulary
        tokens = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]]).unsqueeze(0).to(self.device)

        # For every possible token in maximum length
        for i in range(max_length):
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, hidden, att = self.D(tokens[-1], hidden, outputs)

            #
            atts[i] = att

            # Samples a token using argmax
            sampled_token = preds.argmax(1)

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token.unsqueeze(0)))

            # Check if has reached the end of string
            if sampled_token == trg_field.vocab.stoi[trg_field.eos_token]:
                # If yes, breaks the loop
                break

        # Decodes the tokens into text
        translated_text = [trg_field.vocab.itos[t] for t in tokens]

        return translated_text, atts
