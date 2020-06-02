from torchtext.data import BPTTIterator, Field

from textformer.datasets.generative import GenerativeDataset
from textformer.models import ConvSeq2Seq

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
train_iterator = BPTTIterator(dataset, batch_size=16, bptt_len=10, device=device)

# Creating the ConvSeq2Seq model
conv_seq2seq = ConvSeq2Seq(n_input=len(source.vocab), n_output=len(source.vocab),
                           n_hidden=512, n_embedding=256, n_layers=1, kernel_size=3,
                           ignore_token=None, init_weights=None, device=device)

# Training the model
conv_seq2seq.fit(train_iterator, epochs=10)

# Generating artificial text
text = conv_seq2seq.generate_text(
    'Mr. Dursley', source, length=100, temperature=0.5)

print(' '.join(text))
