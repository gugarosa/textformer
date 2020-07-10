from textformer.models import Transformer

# Creating the Transformer model
transformer = Transformer(n_input=1, n_output=1, n_hidden=512, n_forward=512, n_layers=1,
                          n_heads=3, dropout=0.1, max_length=200, source_pad_index=None,
                          target_pad_index=None, init_weights=None, device='cpu')
