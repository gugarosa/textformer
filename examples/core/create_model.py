from textformer.core.model import Model
from textformer.models.encoders import LSTMEncoder
from textformer.models.decoders import LSTMDecoder

# Creates a template Model class
model = Model(LSTMEncoder(), LSTMDecoder(), ignore_token=None, init_weights=None, device='cpu')
