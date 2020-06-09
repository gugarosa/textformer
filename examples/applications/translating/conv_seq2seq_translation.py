from torchtext.data import BucketIterator, Field

import textformer.utils.visualization as v
from textformer.datasets.translation import TranslationDataset
from textformer.models.conv_seq2seq import ConvSeq2Seq

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
target_pad_index = target.vocab.stoi[target.pad_token]

# Creates a bucket iterator
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), batch_size=2, sort=False, device=device)

# Creating the ConvSeq2Seq model
conv_seq2seq = ConvSeq2Seq(n_input=len(source.vocab), n_output=len(target.vocab),
                           n_hidden=512, n_embedding=256, n_layers=1, kernel_size=3,
                           scale=0.5, max_length=200, ignore_token=target_pad_index,
                           init_weights=None, device=device)

# Training the model
conv_seq2seq.fit(train_iterator, val_iterator, epochs=10)

# Evaluating the model
conv_seq2seq.evaluate(test_iterator)

# Calculating BLEU score
conv_seq2seq.bleu(test_dataset, source, target)
