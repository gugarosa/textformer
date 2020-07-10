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
        """Creates the source mask used in the encoding process.

        Args:
            x (tf.Tensor): Tensor holding the inputs.

        Returns:
            Mask over inputs tensor.

        """
        
        # Creates the encoding mask
        x_mask = (x != self.source_pad_index).unsqueeze(1).unsqueeze(2)

        return x_mask

    def create_target_mask(self, y):
        """Creates the target mask used in the decoding process.

        Args:
            y (tf.Tensor): Tensor holding the targets.

        Returns:
            Mask over targets tensor.
            
        """
        
        # Creates the padded target mask
        y_pad_mask = (y != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        
        # Creates the subtraction target mask
        y_sub_mask = torch.tril(torch.ones((y.shape[1], y.shape[1]))).bool()
        
        # Creates the decoding mask        
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

        # For every possible length
        for i in range(length):
            # Creating encoder mask
            enc_mask = self.create_source_mask(tokens)
            
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Performs the initial encoding
                output = self.E(tokens, enc_mask)

            # Creating decoder mask
            dec_mask = self.create_target_mask(tokens)

            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, _ = self.D(tokens, dec_mask, output, enc_mask)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a token from a categorical distribution based on the predictions
            sampled_token = distributions.Categorical(logits=preds[:, -1].unsqueeze(0)).sample()

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

        # Creating encoder mask
        enc_mask = self.create_source_mask(tokens)

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # Performs the initial encoding
            output = self.E(tokens, enc_mask)

        # Creating a tensor with `<sos>` token from target vocabulary
        tokens = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]]).unsqueeze(0).to(self.device)

        # For every possible token in maximum length
        for i in range(max_length):
            # Creating decoder mask
            dec_mask = self.create_target_mask(tokens)

            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, atts = self.D(tokens, dec_mask, output, enc_mask)

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
