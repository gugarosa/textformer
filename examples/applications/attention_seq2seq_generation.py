from textformer.datasets.generative import GenerativeDataset
from textformer.models.attention_seq2seq import Decoder, Encoder, AttentionSeq2Seq
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
encoder = Encoder(n_input=len(source.vocab), n_hidden_enc=512,
                  n_hidden_dec=512, n_embedding=256)

# Creating the Decoder
decoder = Decoder(n_output=len(source.vocab), n_hidden_enc=512,
                  n_hidden_dec=512, n_embedding=256)

# Creating the AttentionSeq2Seq model
attention_seq2seq = AttentionSeq2Seq(encoder, decoder, init_weights=None,
                                     ignore_token=None, device=device)

# Training the model
attention_seq2seq.fit(train_iterator, epochs=10)

# Generating artificial text
text = attention_seq2seq.sample(source, 'Mr. Dursley', length=100, temperature=0.5)

print(' '.join(text))