import textformer.utils.exception as e
import textformer.utils.logging as l
import torch

logger = l.get_logger(__name__)


class Model(torch.nn.Module):
    """The Model class is the basis for any custom model.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, use_gpu=False):
        """Initialization method.

        Args:
            use_gpu (bool): Whether GPU should be used or not.

        """

        # Override its parent class
        super(Model, self).__init__()

        # Creates a cpu-based device property
        self.device = 'cpu'

        # Checks if GPU is avaliable
        if torch.cuda.is_available() and use_gpu:
            # If yes, change the device property to `cuda`
            self.device = 'cuda'

        # Creating an empty dictionary to hold historical values
        self.history = {}

        # Setting default tensor type to float
        torch.set_default_tensor_type(torch.FloatTensor)

        logger.debug(f'Device: {self.device}.')

    @property
    def device(self):
        """dict: Indicates which device is being used for computation.

        """

        return self._device

    @device.setter
    def device(self, device):
        if device not in ['cpu', 'cuda']:
            raise e.TypeError('`device` should be `cpu` or `cuda`')

        self._device = device

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
        preds = self(x, y, teacher_forcing_ratio=0)

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
        for epoch in range(epochs):
            # Setting the training flag
            self.train()

            # Initializes both losses as zero
            train_loss, val_loss = 0, 0

            # For every batch in the iterator
            for batch in train_iterator:
                # Calculates the training loss
                train_loss += self.step(batch, 1)

            # Gets the mean training loss accross all batches
            train_loss /= len(train_iterator)

            logger.debug(f'Epoch: {epoch+1}/{epochs} | Loss: {train_loss:.4f}')

            # If there is a validation iterator
            if val_iterator:
                # Setting the evalution flag
                self.eval()

                # Inhibits the gradient from updating the parameters
                with torch.no_grad():
                    # For every batch in the iterator
                    for batch in val_iterator:
                        # Calculates the validation loss
                        val_loss += self.val_step(batch)

                # Gets the mean validation loss accross all batches
                val_loss /= len(val_iterator)

                logger.debug(f'Val. Loss: {val_loss:.4f}\n')

    def evaluate(self, test_iterator):
        """Evaluates the model.

        Args:
            test_iterator (torchtext.data.Iterator): Testing data iterator.

        """

        logger.info('Evaluating model ...')

        # Setting the evalution flag
        self.eval()

        # Initializes the loss as zero
        test_loss = 0

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # For every batch in the iterator
            for batch in test_iterator:
                # Calculates the validation loss
                test_loss += self.val_step(batch)

        # Gets the mean validation loss accross all batches
        test_loss /= len(test_iterator)

        logger.debug(f'Loss: {test_loss:.4f}\n')
