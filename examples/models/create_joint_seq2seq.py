from textformer.models import JointSeq2Seq

# Creating the JointSeq2Seq model
joint_seq2seq = JointSeq2Seq(n_input=1, n_output=1, n_hidden=512, n_embedding=256,
                             ignore_token=None, init_weights=None, device='cpu')
