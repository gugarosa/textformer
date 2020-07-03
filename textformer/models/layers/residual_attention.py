import torch
from torch import nn


class ResidualAttention(nn.Module):
    """A ResidualAttention class is used to provide attention-based mechanisms
    in a neural network layer among residual connections.

    References:
        F. Wang, et al. Residual attention network for image classification.
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017).

    """

    def __init__(self, n_hidden, n_embedding, scale):
        """Initialization method.

        Args:
            n_hidden (int): Number of hidden units.
            n_embedding (int): Number of embedding units.
            scale (float): Value for the residual learning.

        """

        # Overriding its parent class
        super(ResidualAttention, self).__init__()

        # Defining the energy-based layer
        self.e = nn.Linear(n_hidden, n_embedding)

        # Defining the weight-based layer
        self.v = nn.Linear(n_embedding, n_hidden)

        # Defining the scale for the residual connections
        self.scale = scale

    def forward(self, emb, c, enc_c, enc_o):
        """Performs a forward pass over the layer.

        Args:
            emb (torch.Tensor): Tensor containing the embedded outputs.
            c (torch.Tensor): Tensor containing the decoder convolutioned features.
            enc_c (torch.Tensor): Tensor containing the encoder convolutioned features.
            enc_o (torch.Tensor): Tensor containing the encoder outputs.

        Returns:
            The attention-based weights, as well as the residual attention-based weights.

        """
        
        # Transforms to the embedding dimension
        emb_c = self.e(c.permute(0, 2, 1))
        
        # Combines the convolutional features and embeddings
        combined = (emb_c + emb) * self.scale
        
        # Calculating the energy between combined and convolutioned states       
        energy = torch.matmul(combined, enc_c.permute(0, 2, 1))

        # Calculating the attention    
        attention = nn.functional.softmax(energy, dim=2)
        
        # Encoding the attention with the combined output
        encoded_attention = torch.matmul(attention, enc_o)
                
        # Converting back to hidden dimension
        encoded_attention = self.v(encoded_attention)
        
        # Applying residual connections
        residual_attention = (c + encoded_attention.permute(0, 2, 1)) * self.scale
        
        return attention, residual_attention
