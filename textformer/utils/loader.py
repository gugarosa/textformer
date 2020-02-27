import textformer.utils.logging as l

logger = l.get_logger(__name__)


def load_text(file_name):
    """Loads a .txt file.

    Args:
        file_name (str): The file name to be loaded.

    Returns:
        A list with the loaded sentences.

    """

    logger.debug(f'Loading {file_name} ...')

    # Tries to load the file
    try:
        # Opens the document file
        file = open(file_name, 'rb')

        # Reads the sentences
        text = file.read().decode(encoding='utf-8').splitlines()

        return text

    # If file can not be loaded
    except FileNotFoundError:
        # Creates an error
        e = f'File not found: {file_name}.'

        # Logs the error
        logger.error(e)

        raise
