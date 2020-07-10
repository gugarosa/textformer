from textformer.models import AttSeq2Seq

# Creating the AttSeq2Seq model
att_seq2seq = AttSeq2Seq(n_input=1, n_output=1, n_hidden_enc=512, n_hidden_dec=512,
                         n_embedding=256, ignore_token=None, init_weights=None, device='cpu')
