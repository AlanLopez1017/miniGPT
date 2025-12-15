import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(Q, K, V, mask = None, dropout = None):
    d_k = Q.size(-1)

    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k) # (B, h, T, dk) @ (B, h, dk, T) -> (B, h, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf')) # (B, h, T, T)
    
    p_attn = F.softmax(scores, dim = -1) # (B, h, T, T)

    if dropout is not None:
        p_attn = dropout(p_attn) # (B, h, T, T)
    
    out = p_attn @ V # (B, h, T, T) @ (B, h, T, dk) -> (B, h, T, dk)

    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()

        assert n_embd // num_heads

        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd

        self.key = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.query = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.value = nn.Linear(n_embd, head_size * num_heads, bias=False)

        # Mask (lower-triangular)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B: Batch, T: Time (Block_size), C: Channels (n_embd)
        B, T, C = x.shape

        # linear projections
        q = self.query(x) # (B, T, num_heads*head_size = n_embd)
        k = self.key(x)
        v = self.value(x)

        # Change to (B, num_heads = h, T = block_size, head_size = dk)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        mask = self.tril[:T, :T].unsqueeze(0).unsqueeze(0) # (1, 1, T, T)

        # Scaled dot-product attention
        attn_out = attention(q, k, v, mask, self.dropout) # (B, h, T, dk)

        # To (B, T, n_embd)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, self.num_heads*self.head_size)

        # Final projection and dropout
        out = self.proj(attn_out)
        out = self.dropout(out)

        return out













        
        