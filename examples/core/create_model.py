from textformer.core.model import Model
from textformer.models.decoders import LSTMDecoder
from textformer.models.encoders import LSTMEncoder

# Creates a template Model class
model = Model(LSTMEncoder(), LSTMDecoder(), ignore_token=None, init_weights=None, device='cpu')
