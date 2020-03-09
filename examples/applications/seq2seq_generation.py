from textformer.datasets.generative import GenerativeDataset
from textformer.models.seq2seq import Decoder, Encoder, Seq2Seq
from torchtext.data import BPTTIterator, Field

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

# Defines the input file
file_path = 'data/generative/chapter1_harry.txt'

# Defines a datatype for further tensor conversion
source = Field(lower=True)

# Creates the GenerativeDataset
dataset = GenerativeDataset(file_path, source)

# Builds the vocabulary
source.build_vocab(dataset, min_freq=1)

# Creates an iterator that backpropagates through time
train_iterator = BPTTIterator(
    dataset, batch_size=16, bptt_len=10, device=device)

# Creating the Encoder
encoder = Encoder(n_input=len(source.vocab), n_hidden=512,
                  n_embedding=256, n_layers=2)

# Creating the Decoder
decoder = Decoder(n_output=len(source.vocab), n_hidden=512,
                  n_embedding=256, n_layers=2)

# Creating the Seq2Seq model
seq2seq = Seq2Seq(encoder, decoder, init_weights=None,
                  ignore_token=None, device=device)

# Training the model
seq2seq.fit(train_iterator, epochs=50)

# Generating artificial text
text = seq2seq.sample(source, 'Mr. Dursley', length=100, temperature=0.5)

print(' '.join(text))