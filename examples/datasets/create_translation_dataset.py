from torchtext.data import BucketIterator, Field

from textformer.datasets.translation import TranslationDataset

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

# Defines the input file
file_path = 'data/translation/europarl'

# Defines datatypes for further tensor conversion
source = Field(init_token='<sos>', eos_token='<eos>', lower=True)
target = Field(init_token='<sos>', eos_token='<eos>', lower=True)

# Creates the TranslationDataset
train_dataset, val_dataset, test_dataset = TranslationDataset.splits(
    file_path, ('.en', '.pt'), (source, target))

# Builds the vocabularies
source.build_vocab(train_dataset, min_freq=1)
target.build_vocab(train_dataset, min_freq=1)

# Creates a bucket iterator
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), batch_size=16, sort=False, device=device)
