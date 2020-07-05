from torchtext.data import BucketIterator, Field

from textformer.datasets.translation import TranslationDataset
from textformer.models.transformer import Transformer

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

# Defines the input file
file_path = 'data/translation/europarl'

# Defines datatypes for further tensor conversion
source = Field(init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
target = Field(init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

# Creates the TranslationDataset
train_dataset, val_dataset, test_dataset = TranslationDataset.splits(
    file_path, ('.en', '.pt'), (source, target))

# Builds the vocabularies
source.build_vocab(train_dataset, min_freq=1)
target.build_vocab(train_dataset, min_freq=1)

# Gathering the <pad> token index for further ignoring
source_pad_index = target.vocab.stoi[source.pad_token]
target_pad_index = target.vocab.stoi[target.pad_token]

# Creates a bucket iterator
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), batch_size=2, sort=False, device=device)

# Creating the Transformer model
transformer = Transformer(n_input=len(source.vocab), n_output=len(target.vocab),
                          n_hidden=512, n_forward=512, n_layers=1, n_heads=3,
                          dropout=0.1, max_length=200, source_pad_index=source_pad_index,
                          target_pad_index=target_pad_index, init_weights=None, device=device)

# Training the model
transformer.fit(train_iterator, val_iterator, epochs=1)

# Evaluating the model
transformer.evaluate(test_iterator)

# Calculating BLEU score
transformer.bleu(test_dataset, source, target)
