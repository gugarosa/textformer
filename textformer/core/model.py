import math
import time

import torch
from torch import nn, optim
from tqdm import tqdm

import textformer.utils.exception as e
import textformer.utils.logging as l

logger = l.get_logger(__name__)


class Encoder(torch.nn.Module):
    """An Encoder class is responsible for easily-implementing the encoding part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Encoder, self).__init__()

    def forward(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (torch.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


class Decoder(torch.nn.Module):
    """A Decoder class is responsible for easily-implementing the decoding part of
    a neural network, when custom training or additional sets are not needed.

    """

    def __init__(self):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Decoder, self).__init__()

    def forward(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (torch.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


class Model(torch.nn.Module):
    """A Model class is responsible for customly implementing Sequence-To-Sequence and Transformer architectures.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, encoder, decoder, ignore_token=None, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            encoder (Encoder): Network's encoder architecture.
            decoder (Decoder): Network's decoder architecture.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            ignore_token (int): The index of a token to be ignore by the loss function.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(Model, self).__init__()

        # Defining the encoder network
        self.E = encoder

        # Defining the decoder network
        self.D = decoder

        # Creating an empty dictionary to hold historical values
        self.history = {}

        # Creates a cpu-based device property
        self.device = device

        # Compiles the network's additional properties
        self._compile(ignore_token, init_weights)

        # Checks if GPU is avaliable
        if torch.cuda.is_available() and device == 'cuda':
            # Uses CUDA in the whole class
            self.cuda()

        # Setting default tensor type to float
        torch.set_default_tensor_type(torch.FloatTensor)

        logger.debug(f'Device: {self.device}.')

    @property
    def E(self):
        """Encoder: Encoder architecture.

        """

        return self._E

    @E.setter
    def E(self, E):
        if not isinstance(E, Encoder):
            raise e.TypeError('`E` should be a Encoder')

        self._E = E

    @property
    def D(self):
        """Decoder: Decoder architecture.

        """

        return self._D

    @D.setter
    def D(self, D):
        if not isinstance(D, Decoder):
            raise e.TypeError('`D` should be a Decoder')

        self._D = D

    @property
    def history(self):
        """dict: Dictionary containing historical values from the model.

        """

        return self._history

    @history.setter
    def history(self, history):
        if not isinstance(history, dict):
            raise e.TypeError('`history` should be a dictionary')

        self._history = history

    @property
    def device(self):
        """str: Indicates which device is being used for computation.

        """

        return self._device

    @device.setter
    def device(self, device):
        if device not in ['cpu', 'cuda']:
            raise e.TypeError('`device` should be `cpu` or `cuda`')

        self._device = device

    def _compile(self, ignore_token, init_weights):
        """Compiles the network by setting its optimizer, loss function and additional properties.

        """

        # Defining an optimizer
        self.optimizer = optim.Adam(self.parameters())

        # Checking if there is a token to be ignored
        if ignore_token:
            # If yes, define loss based on it
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_token)

        # If there is no token to be ignored
        else:
            # Defines the loss as usual
            self.loss = nn.CrossEntropyLoss()

        # Check if there is a tuple for the weights initialization
        if init_weights:
            # Iterate over all possible parameters
            for _, p in self.named_parameters():
                # Initializes with a uniform distributed value
                nn.init.uniform_(p.data, init_weights[0], init_weights[1])

    def dump(self, **kwargs):
        """Dumps any amount of keyword documents to lists in the history property.

        """

        # Iterate through key-word arguments
        for k, v in kwargs.items():
            # Check if there is already an instance of current
            if k not in self.history.keys():
                # If not, creates an empty list
                self.history[k] = []

            # Appends the new value to the list
            self.history[k].append(v)

    def step(self, batch, clip):
        """Performs a single batch optimization step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).
            clip (float): Value to clip the gradients.

        Returns:
            The training loss accross the batch.

        """

        # Gathers the batch's input and target
        x, y = batch.text, batch.target

        # Resetting the gradients
        self.optimizer.zero_grad()

        # Calculate the predictions based on inputs
        preds = self(x, y)

        # Reshaping the tensor's size without the batch dimension
        preds = preds[1:].view(-1, preds.shape[-1])

        # Reshaping the tensor's size without the batch dimension
        y = y[1:].view(-1)

        # Calculates the batch's loss
        batch_loss = self.loss(preds, y)

        # Propagates the gradients backward
        batch_loss.backward()

        # Clips the gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        # Perform the parameeters updates
        self.optimizer.step()

        return batch_loss.item()

    def val_step(self, batch):
        """Performs a single batch evaluation step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).

        Returns:
            The validation loss accross the batch.

        """

        # Gathers the batch's input and target
        x, y = batch.text, batch.target

        # Calculate the predictions based on inputs
        preds = self(x, y, teacher_forcing_ratio=0.0)

        # Reshaping the tensor's size without the batch dimension
        preds = preds[1:].view(-1, preds.shape[-1])

        # Reshaping the tensor's size without the batch dimension
        y = y[1:].view(-1)

        # Calculates the batch's loss
        batch_loss = self.loss(preds, y)

        return batch_loss.item()

    def fit(self, train_iterator, val_iterator=None, epochs=10):
        """Trains the model.

        Args:
            train_iterator (torchtext.data.Iterator): Training data iterator.
            val_iterator (torchtext.data.Iterator): Validation data iterator.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Setting the training flag
            self.train()

            # Initializes both losses as zero
            train_loss, val_loss = 0.0, 0.0

            # Defines a `tqdm` variable
            with tqdm(total=len(train_iterator)) as t:
                # For every batch in the iterator
                for i, batch in enumerate(train_iterator):
                    # Calculates the training loss
                    train_loss += self.step(batch, 1)
                     
                    # Updates the `tqdm` status
                    t.set_postfix(loss=train_loss / (i + 1))
                    t.update()

            # Gets the mean training loss accross all batches
            train_loss /= len(train_iterator)

            logger.info(f'Loss: {train_loss} | PPL: {math.exp(train_loss)}')

            # If there is a validation iterator
            if val_iterator:
                # Setting the evalution flag
                self.eval()

                # Inhibits the gradient from updating the parameters
                with torch.no_grad():
                    # Defines a `tqdm` variable
                    with tqdm(total=len(train_iterator)) as t:
                        # For every batch in the iterator
                        for i, batch in enumerate(val_iterator):
                            # Calculates the validation loss
                            val_loss += self.val_step(batch)

                            # Updates the `tqdm` status
                            t.set_postfix(val_loss=val_loss / (i + 1))
                            t.update()

                # Gets the mean validation loss accross all batches
                val_loss /= len(val_iterator)

                logger.info(f'Val Loss: {val_loss} | Val PPL: {math.exp(val_loss)}')

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(loss=train_loss, val_loss=val_loss, time=end-start)

    def evaluate(self, test_iterator):
        """Evaluates the model.

        Args:
            test_iterator (torchtext.data.Iterator): Testing data iterator.

        """

        logger.info('Evaluating model ...')

        # Setting the evalution flag
        self.eval()

        # Initializes the loss as zero
        test_loss = 0.0

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # For every batch in the iterator
            for i, batch in enumerate(test_iterator):
                # Calculates the test loss
                test_loss += self.val_step(batch)

        # Gets the mean validation loss accross all batches
        test_loss /= len(test_iterator)

        logger.info(f'Loss: {test_loss} | PPL: {math.exp(test_loss)}')

    def generate_text(self, start, field, length=10, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own sample (text generation) implementation.

        Args:
            start (str): The start string to generate the text.
            field (torchtext.data.Field): Datatype instructions for tensor convertion.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def translate_text(self, start, src_field, trg_field, max_length=10):
        """Translates text from the source vocabulary to the target vocabulary.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own translation implementation.

        Args:
            start (str): The string to be translated.
            src_field (torchtext.data.Field): Source vocabulary datatype instructions for tensor convertion.
            trg_field (torchtext.data.Field): Target vocabulary datatype instructions for tensor convertion.
            max_length (int): Maximum length of translated text.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

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

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
