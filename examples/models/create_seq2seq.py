from textformer.models import Seq2Seq

# Creating the Seq2Seq model
seq2seq = Seq2Seq(n_input=1, n_output=1, n_hidden=512, n_embedding=256, n_layers=2,
                  ignore_token=None, init_weights=None, device='cpu')
