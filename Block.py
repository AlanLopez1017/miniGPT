import torch.nn as nn
from Attention2 import *
from PositionWiseFeedForward import *

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout = 0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = PWFeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Pre-Norm and Residual for MHA
        x = x + self.sa(self.ln1(x))
        # Pre-Norm and Residual for MLP
        x = x + self.ffwd(self.ln2(x))

        return x


