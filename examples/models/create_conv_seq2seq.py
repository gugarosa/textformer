from textformer.models import ConvSeq2Seq

# Creating the ConvSeq2Seq model
conv_seq2seq = ConvSeq2Seq(n_input=1, n_output=1, n_hidden=512, n_embedding=256, n_layers=1,
                           kernel_size=3, scale=0.5, ignore_token=None, init_weights=None, device='cpu')
