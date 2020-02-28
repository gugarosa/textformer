from textformer.datasets.generative import GenerativeDataset
from torchtext.data import BPTTIterator, Field

# Defines the input file
file_path = 'data/generative/chapter1_harry.txt'

# Defines a datatype for further tensor conversion
field = Field(batch_first=True, lower=True)

# Creates the GenerativeDataset
dataset = GenerativeDataset(file_path, field)

# Builds the vocabulary
field.build_vocab(dataset, min_freq=1)

# Creates an iterator that backpropagates through time
iterator = BPTTIterator(dataset, batch_size=16, bptt_len=10)

for i in iterator:
    print(i.text, i.target)
