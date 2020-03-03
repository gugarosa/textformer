from textformer.datasets.generative import GenerativeDataset
from textformer.models.seq2seq import Decoder, Encoder, Seq2Seq
from torchtext.data import BPTTIterator, Field

# Defines the input file
file_path = 'data/generative/chapter1_harry.txt'

# Defines a datatype for further tensor conversion
source = Field(batch_first=True, lower=True)

# Creates the GenerativeDataset
dataset = GenerativeDataset(file_path, field)

# Builds the vocabulary
source.build_vocab(dataset, min_freq=1)

# Creates an iterator that backpropagates through time
train_iterator = BPTTIterator(dataset, batch_size=16, bptt_len=10)

# Creating the Encoder
encoder = Encoder(n_input=len(source.vocab), n_hidden=512,
                  n_embedding=256, n_layers=2)

# Creating the Decoder
decoder = Decoder(n_output=len(source.vocab), n_hidden=512,
                  n_embedding=256, n_layers=2)

# Creating the Seq2Seq model
seq2seq = Seq2Seq(encoder, decoder)

# Training the model
seq2seq.fit(train_iterator, epochs=10)
