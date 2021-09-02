"""Text generation dataset.
"""

import io

import torchtext.data as data

import textformer.utils.logging as l

logger = l.get_logger(__name__)


class GenerativeDataset(data.Dataset):
    """A GenerativeDataset class is in charge of loading raw texts and creating
    Language Modelling datasets, used for text generation tasks.

    """

    def __init__(self, file_path, field, **kwargs):
        """Creates a GenerativeDataset, used for text generation.

        Args:
            file_path (str): Path to the file that will be loaded.
            field (torchtext.data.Field): Datatype instructions for tensor convertion.

        """

        logger.info('Overriding class: torchtext.data.Dataset -> GenerativeDataset.')

        # Creates a `text` field from the input field
        fields = [('text', field)]

        # Loads the input file and creates a list of examples
        example = self._load_data(file_path, fields)

        super(GenerativeDataset, self).__init__(example, fields, **kwargs)

        logger.info('Class overrided.')

    def _load_data(self, file_path, fields):
        """Loads a text file and creates a list of torchtext Example classes.

        Args:
            file_path (str): Path to the file that will be loaded.
            fields (list): List of tuples holding datatype instructions for tensor convertion.

        Returns:
            The loaded and pre-processed text within a list of Example classes.

        """

        logger.debug('Loading %s ...', file_path)

        try:
            with io.open(file_path, mode='r', encoding='utf-8') as f:
                text = []

                for line in f:
                    text += fields[0][1].preprocess(line)

                logger.debug('Data loaded.')

            # Creates a list of examples based on loaded text and pre-defined field
            example = [data.Example.fromlist([text], fields)]

            return example

        except FileNotFoundError:
            e = f'File not found: {file_path}.'

            logger.error(e)

            raise
