import io

from torchtext import data


class GenerativeDataset(data.Dataset):
    """
    """

    def __init__(self, file_path, field, **kwargs):
        """Creates a GenerativeDataset, used for text generation.

        Args:

        """

        #
        fields = [('text', field)]

        #
        examples = []

        source = []
        
        #
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            #
            for line in f:
                #
                source += field.preprocess(line)

                #
        examples.append(data.Example.fromlist([source], fields))

        #
        super(GenerativeDataset, self).__init__(examples, fields, **kwargs)
