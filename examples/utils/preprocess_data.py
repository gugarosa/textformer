from torchtext.data import Pipeline

import textformer.utils.preprocess as p

# Defines an input text
s = 'Text to be pre-processed!'

# Creates a pre-processing pipeline
pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_char)

# Adds to the torchtext Pipeline
p = Pipeline(pipe)

# Pre-process the input text
tokens = p.call(s)

print(tokens)
