import torch

import textformer.utils.exception as e
import textformer.utils.logging as l

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
            x (): A tensor containing the inputs.
            y (): A tensor containing the inputs' labels.
            clip ():

        """

        #
        source, target = batch.source, batch.target

        #
        output = self(source, target)

        #
        output_size = output.shape[-1]

        #
        output = output[1:].view(-1, output_size)

        #
        target = target[1:].view(-1)

        #
        batch_loss = self.loss(output, target)

        #
        batch_loss.backward()

        #
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        #
        self.optimizer.step()

        return batch_loss.item()


    # def step(self, iterator, clip):
    #     """
    #     """

    #     #
    #     self.train()

    #     #
    #     loss = 0

    #     #
    #     for i, batch in enumerate(iterator):
    #         #
    #         source, target = batch.source, batch.target

    #         #
    #         self.optimizer.zero_grad()

    #         #
    #         output = self(source, target)

    #         #
    #         output_size = output.shape[-1]

    #         #
    #         output = output[1:].view(-1, output_size)

    #         #
    #         target = target[1:].view(-1)

    #         #
    #         batch_loss = self.loss(output, target)

    #         #
    #         batch_loss.backward()

    #         #
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

    #         #
    #         self.optimizer.step()

    #         #
    #         loss += batch_loss.item()

    #     return loss / len(iterator)

    def val_step(self, batch):
        """
        """

        #
        source, target = batch.source, batch.target

        #
        output = self(source, target, 0)

        #
        output_size = output.shape[-1]

        #
        output = output[1:].view(-1, output_size)

        #
        target = target[1:].view(-1)

        #
        batch_loss = self.loss(output, target)

        return batch_loss.item()
                

    def fit(self, train_iterator, val_iterator=None, epochs=10):
        """Trains the model.

        Args:

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for epoch in range(epochs):
            #
            self.train()

            #
            train_loss = 0
            val_loss = 0

            #
            for batch in train_iterator:
                #
                train_loss += self.step(batch, 1)

            #
            train_loss /= len(train_iterator)

            logger.debug(
                f'Epoch: {epoch+1}/{epochs} | Loss: {train_loss:.4f}')

            #
            self.eval()
            
            #
            if val_iterator:
                #
                with torch.no_grad():
                    #
                    for batch in val_iterator:
                        #
                        val_loss += self.val_step(batch)
                    
            #
            val_loss /= len(val_iterator)

            logger.debug(
                    f'Val Loss: {val_loss:.4f}\n')

