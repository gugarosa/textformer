import io

from torchtext import data
import textformer.utils.logging as l

logger = l.get_logger(__name__)


class GenerativeDataset(data.Dataset):
    """
    """

    def __init__(self, file_path, field, **kwargs):
        """Creates a GenerativeDataset, used for text generation.

        Args:

        """

        logger.info('Overriding class: torchtext.data.Dataset -> GenerativeDataset.')

        #
        fields = [('text', field)]

        #
        text = self._load_text(file_path, field)

        #
        example = [data.Example.fromlist([text], fields)]

        #
        super(GenerativeDataset, self).__init__(example, fields, **kwargs)

        logger.info('Class overrided.')

    def _load_text(self, file_path, field):
        """
        """

        logger.debug(f'Loading {file_path} ...')

        #
        try:
            #
            with io.open(file_path, mode='r', encoding='utf-8') as f:
                #
                text = []

                #
                for line in f:
                    #
                    text += field.preprocess(line)

                logger.debug(f'Data loaded.')

                return text
        
        # If file can not be loaded
        except FileNotFoundError:
            # Creates an error
            e = f'File not found: {file_path}.'

            # Logs the error
            logger.error(e)

            raise
            


        
