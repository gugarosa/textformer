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
            conv, output = self.E(tokens)

        # For every possible length
        for i in range(length):
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, _ = self.D(tokens[:,-1].unsqueeze(0), conv, output)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a token from a categorical distribution based on the predictions
            sampled_token = distributions.Categorical(logits=preds).sample()

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token), axis=1)

        # Decodes the tokens into text
        sampled_text = [field.vocab.itos[t] for t in tokens.squeeze(0)]

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

        # Setting the evalution flag
        self.eval()

        # Pre-processing the start text into tokens
        tokens = src_field.preprocess(start)

        # Adding `<sos>`` and `<eos>` to the tokens
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        # Numericalizing the tokens
        tokens = src_field.numericalize([tokens]).to(self.device)

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # Performs the initial encoding
            conv, output = self.E(tokens)

        # Creating a tensor with `<sos>` token from target vocabulary
        tokens = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]]).unsqueeze(0).to(self.device)

        # For every possible token in maximum length
        for i in range(max_length):
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, atts = self.D(tokens, conv, output)

            # Samples a token using argmax
            sampled_token = preds.argmax(2)[:,-1]

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token.unsqueeze(0)), axis=1)

            # Check if has reached the end of string
            if sampled_token == trg_field.vocab.stoi[trg_field.eos_token]:
                # If yes, breaks the loop
                break

        # Decodes the tokens into text
        translated_text = [trg_field.vocab.itos[t] for t in tokens.squeeze(0)]

        return translated_text[1:], atts

    def bleu(self, dataset, src_field, trg_field, max_length=50, n_grams=4):
        """Calculates BLEU score over a dataset from its difference between targets and predictions.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own bleu implementation, due to having different translation methods.

        Args:
            dataset (torchtext.data.Dataset): Dataset to have its BLEU calculated.
            src_field (torchtext.data.Field): Source vocabulary datatype instructions for tensor convertion.
            trg_field (torchtext.data.Field): Target vocabulary datatype instructions for tensor convertion.
            max_length (int): Maximum length of translated text.
            n_grams (int): Maxmimum n-grams to be used.

        Returns:
            BLEU score from input dataset.

        """

        logger.info(f'Calculating BLEU with {n_grams}-grams ...')

        # Defines a list for holding the targets and predictions
        targets, preds = [], []

        # For every example in the dataset
        for data in dataset:
            # Calculates the prediction, i.e., translated text
            pred, _ = self.translate_text(data.text, src_field, trg_field, max_length)

            # Appends the prediction without the `<eos>` token
            preds.append(pred[:-1])

            # Appends an iterable of the target
            targets.append([data.target])

        # Calculates the BLEU score
        bleu = bleu_score(preds, targets, max_n=n_grams)

        logger.info(f'BLEU: {bleu}')

        return bleu
