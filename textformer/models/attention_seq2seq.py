import torch
from torch import distributions, nn, optim

import textformer.utils.logging as l
from textformer.core.model import Model
from textformer.models.layers.attention import Attention

logger = l.get_logger(__name__)


class Encoder(nn.Module):
    """An Encoder class is used to supply the encoding part of the AttentionSeq2Seq architecture.

    """

    def __init__(self, n_input=128, n_hidden_enc=128, n_hidden_dec=128, n_embedding=128, dropout=0.5):
        """Initializion method.

        Args:
            n_input (int): Number of input units.
            n_hidden_enc (int): Number of hidden units in the Encoder.
            n_hidden_dec (int): Number of hidden units in the Decoder.
            n_embedding (int): Number of embedding units.
            dropout (float): Amount of dropout to be applied.

        """

        logger.info('Creating class: Encoder.')

        # Overriding its parent class
        super(Encoder, self).__init__()

        # Number of input units
        self.n_input = n_input

        # Number of hidden units
        self.n_hidden = n_hidden_enc

        # Number of embedding units
        self.n_embedding = n_embedding

        # Embedding layer
        self.embedding = nn.Embedding(n_input, n_embedding)

        # RNN layer
        self.rnn = nn.GRU(n_embedding, n_hidden_enc, bidirectional=True)

        # Fully-connected layer
        self.fc = nn.Linear(n_hidden_enc * 2, n_hidden_dec)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f'Size: ({self.n_input}, {self.n_hidden}) | Embeddings: {self.n_embedding} | Core: {self.rnn} | Output: {self.fc}.')

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The hidden state and cell values.

        """

        # Calculates the embedded layer outputs
        embedded = self.dropout(self.embedding(x))

        # Calculates the RNN outputs
        outputs, hidden = self.rnn(embedded)

        # Initial Decoder hidden layer is the final hidden state of the Encoder forward and backward RNNs
        # Also, they are fed through a Linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        return outputs, hidden


class Decoder(nn.Module):
    """A Decoder class is used to supply the decoding part of the AttentionSeq2Seq architecture.

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

        logger.info('Creating class: Decoder.')

        # Overriding its parent class
        super(Decoder, self).__init__()

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

    def forward(self, x, h, y):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.
            h (torch.Tensor): Tensor containing the hidden states.
            y (torch.Tensor): Tensor containing the encoder outputs.

        Returns:
            The prediction and hidden state.

        """

        # Calculates the embedded layer
        embedded = self.dropout(self.embedding(x.unsqueeze(0)))

        # Calculates the attention
        attention = self.a(h, y).unsqueeze(1)

        # Permutes the encoder outputs
        encoder_outputs = y.permute(1, 0, 2)

        # Calculates the weights from the attention-based layer
        weighted = torch.bmm(attention, encoder_outputs).permute(1, 0, 2)

        # Calculates the RNN layer
        output, hidden = self.rnn(torch.cat((embedded, weighted), dim=2), h.unsqueeze(0))

        # Concatenating the output with hidden and context tensors
        output = torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1)

        # Calculates the prediction over the fully connected layer
        pred = self.fc(output)

        return pred, hidden.squeeze(0)


class AttentionSeq2Seq(Model):
    """An AttentionSeq2Seq class implements an attention-based Sequence to Sequence learning architecture.

    """

    def __init__(self, encoder, decoder, init_weights=None, ignore_token=None, device='cpu'):
        """Initialization method.

        Args:
            encoder (Encoder): An Encoder object.
            decoder (Decoder): A Decoder object.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            ignore_token (int): The index of a token to be ignore by the loss function.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        logger.info('Overriding class: Model -> AttentionSeq2Seq.')

        # Overriding its parent class
        super(AttentionSeq2Seq, self).__init__(device=device)

        # Applying the encoder as a property
        self.encoder = encoder

        # Applying the decoder as a property
        self.decoder = decoder

        # Defining an optimizer
        self.optimizer = optim.Adam(self.parameters())

        # Checking if there is a token to be ignore
        if ignore_token:
            # If yes, define loss based on it
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_token)
        
        # If there is no token to be ignored
        else:
            # Defines the loss as usual
            self.loss = nn.CrossEntropyLoss()

        # Check if there is a variable for the weights initialization
        if init_weights:
            # Iterate over all possible parameters
            for _, p in self.named_parameters():
                # Initializes with a uniform distributed value
                nn.init.uniform_(p.data, init_weights[0], init_weights[1])

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

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
        preds = torch.zeros(y.shape[0], y.shape[1],
                            self.decoder.n_output, device=self.device)

        # Performs the initial encoding
        outputs, hidden = self.encoder(x)

        # Make sure that the first decoding will come from the true labels
        x = y[0, :]

        # For every possible token in the sequence
        for t in range(1, y.shape[0]):
            # Decodes the tensor
            pred, hidden = self.decoder(x, hidden, outputs)

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
            outputs, hidden = self.encoder(tokens)

        # Removes the batch dimension from the tokens
        tokens = tokens.squeeze(0)

        # For every possible length
        for i in range(length):
            # Inhibits the gradient from updating the parameters
            with torch.no_grad():
                # Decodes only the last token, i.e., last sampled token
                preds, hidden = self.decoder(tokens[-1], hidden, outputs)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a token from a categorical distribution based on the predictions
            sampled_token = distributions.Categorical(logits=preds).sample()

            # Concatenate the sampled token with the input tokens
            tokens = torch.cat((tokens, sampled_token.unsqueeze(0)))

        # Decodes the tokens into text
        sampled_text = [field.vocab.itos[t] for t in tokens]

        return sampled_text
