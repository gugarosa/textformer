from textformer.datasets.translation import TranslationDataset
from torchtext.data import BucketIterator, Field

# Defines the input file
file_path = 'data/translation/sample.txt'

# Defines datatypes for further tensor conversion
source = Field(init_token='<sos>', eos_token='<eos>', lower=True)
target = Field(init_token='<sos>', eos_token='<eos>', lower=True)

# Creates the TranslationDataset
dataset = TranslationDataset(file_path, ('.en', '.pt'), (source, target))

# Builds the vocabularies
source.build_vocab(dataset, min_freq=1)
target.build_vocab(dataset, min_freq=1)

# Creates a bucket iterator
iterator = BucketIterator(dataset, batch_size=2)

for i in iterator:
    print(i.source, i.target)
