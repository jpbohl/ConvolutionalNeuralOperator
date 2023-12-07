import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """Self attention Layer."""

    def __init__(self, in_dim, n, k):
        """
        :param in_dim: The number of input channels
        :param n: The total number of grid points
        :param k: The number of gridpoints to reduce to
        """

        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        k = in_dim

        R = torch.randn(n, k) / k
        delta = 1 / n
        self.E = R * delta
        self.F = R * torch.exp(-delta)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): Input feature maps (B X C X W X H).

        Returns:
            out (torch.Tensor): Self-attention value + input feature.
            attention (torch.Tensor): Attention weights (B X N X N, where N is Width*Height).
        """
        m_batchsize, C, width, height = x.size()
        
        # Compute Query and Key Convolutions
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # Reduce Key Size
        proj_key = torch.bmm(self.E, proj_key)

        # Compute self similarity
        attention = torch.bmm(proj_query, proj_key)
        
        del proj_query
        del proj_key

        attention = self.softmax(attention)
        
        # Compute value convolution and reduce size
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        proj_value = torch.bmm(self.F, proj_value)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out