from textformer.datasets.translation import TranslationDataset
from textformer.models.seq2seq import Encoder, Decoder, Seq2Seq
from torchtext.data import BucketIterator, Field

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
    (train_dataset, val_dataset, test_dataset), batch_size=2, sort=False)

# Creating the Encoder
encoder = Encoder(n_input=len(source.vocab), n_hidden=512, n_embedding=256, n_layers=2)

# Creating the Decoder
decoder = Decoder(n_output=len(target.vocab), n_hidden=512, n_embedding=256, n_layers=2)

# Creating the Seq2Seq model
seq2seq = Seq2Seq(encoder, decoder)

# Training the model
seq2seq.fit(train_iterator, val_iterator, 10)
