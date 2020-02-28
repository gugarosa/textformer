import os
import io

import textformer.utils.logging as l
from torchtext import data

logger = l.get_logger(__name__)


class TranslationDataset(data.Dataset):
    """A TranslationDataset class is in charge of loading (source, target) texts and creating
    Machine Translation datasets, used for translating tasks.

    """

    def __init__(self, file_path, extensions, fields, **kwargs):
        """Creates a TranslationDataset, used for text translation.

        Args:
            file_path (str): Path to the file that will be loaded.
            extensions (tuple): Extensions to the path for each language.
            fields (tuple): Tuple of datatype instructions for tensor convertion.

        """

        logger.info(
            'Overriding class: torchtext.data.Dataset -> TranslationDataset.')

        # Creates `source` and `target` fields from the input field
        fields = [('source', fields[0]), ('target', fields[1])]

        # Extending file's path with extensions
        source_path, target_path = tuple(
            os.path.expanduser(file_path + e) for e in extensions)

        # Loads the input file and creates a list of examples
        examples = self._load_data(source_path, target_path, fields)

        # Overriding its parent class
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

        logger.info('Class overrided.')

    def _load_data(self, source_path, target_path, fields):
        """Loads a .txt file and creates a list of torchtext Example classes.

        Args:
            source_path (str): Path to the source file that will be loaded.
            target_path (str): Path to the target file that will be loaded.
            fields (tuple): Tuple of datatype instructions for tensor convertion.

        Returns:
            The loaded and pre-processed source and target within a list of Example classes.

        """

        logger.debug(f'Loading {source_path} and {target_path} ...')

        # Tries to invoke the following functions
        try:
            # Creates a list to hold the examples
            examples = []

            # While both files are open
            with io.open(source_path, mode='r', encoding='utf-8') as s, io.open(target_path, mode='r', encoding='utf-8') as t:
                # For every line in both files
                for source_line, target_line in zip(s, t):
                    # Strips the line and adds back to the variable
                    source_line, target_line = source_line.strip(), target_line.strip()

                    # Checks if both lines have something
                    if source_line != '' and target_line != '':
                        # Appends to the list an example based on loaded source and target and pre-defined fields
                        examples.append(data.Example.fromlist(
                            [source_line, target_line], fields))

                logger.debug(f'Data loaded.')

            return examples

        # If file can not be loaded
        except FileNotFoundError:
            # Creates an error
            e = f'File not found: {file_path}.'

            # Logs the error
            logger.error(e)

            raise
