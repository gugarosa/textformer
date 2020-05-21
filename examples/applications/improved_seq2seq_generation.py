from torchtext.data import BPTTIterator, Field

from textformer.datasets.generative import GenerativeDataset
from textformer.models import ImprovedSeq2Seq

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

# Defines the input file
file_path = 'data/generative/chapter1_harry.txt'

# Defines a datatype for further tensor conversion (word tokenization)
source = Field(lower=True)

# Defines a datatype for further tensor conversion (char tokenization)
# source = Field(lower=True, tokenize=lambda x: list(x))

# Creates the GenerativeDataset
dataset = GenerativeDataset(file_path, source)

# Builds the vocabulary
source.build_vocab(dataset, min_freq=1)

# Creates an iterator that backpropagates through time
train_iterator = BPTTIterator(dataset, batch_size=16, bptt_len=10, device=device)

# Creating the ImprovedSeq2Seq model
improved_seq2seq = ImprovedSeq2Seq(n_input=len(source.vocab), n_output=len(source.vocab),
                                   n_hidden=512, n_embedding=256, ignore_token=None,
                                   init_weights=None, device=device)

# Training the model
improved_seq2seq.fit(train_iterator, epochs=25)

# Generating artificial text
text = improved_seq2seq.sample(
    source, 'Mr. Dursley', length=100, temperature=0.5)

print(' '.join(text))
