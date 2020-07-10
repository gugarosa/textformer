from torchtext.data import BPTTIterator, Field

from textformer.datasets.generative import GenerativeDataset
from textformer.models.transformer import Transformer

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

# Defines the input file
file_path = 'data/generative/chapter1_harry.txt'

# Defines a datatype for further tensor conversion
source = Field(lower=True, batch_first=True)

# Creates the GenerativeDataset
dataset = GenerativeDataset(file_path, source)

# Builds the vocabulary
source.build_vocab(dataset, min_freq=1)

# Gathering the <pad> token index for further ignoring
pad_index = source.vocab.stoi[source.pad_token]

# Creates an iterator that backpropagates through time
train_iterator = BPTTIterator(dataset, batch_size=16, bptt_len=10, device=device)

# Creating the Transformer model
transformer = Transformer(n_input=len(source.vocab), n_output=len(source.vocab),
                          n_hidden=512, n_forward=512, n_layers=1, n_heads=3,
                          dropout=0.1, max_length=200, source_pad_index=pad_index,
                          target_pad_index=pad_index, init_weights=None, device=device)

# Training the model
transformer.fit(train_iterator, epochs=10)

# Generating artificial text
text = transformer.generate_text('Mr. Dursley', source, length=10, temperature=0.9)

print(' '.join(text))
