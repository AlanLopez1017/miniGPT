import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int): # dropout
        super().__init__() # super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        den = torch.exp(
             torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model) # float
        )

        pe[:, 0::2] = torch.sin(position * den) 
        pe[:, 1::2] = torch.cos(position * den)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x # self.dropout(x)

def plot_pe(max_len = 100, d_model = 64):
    import matplotlib.pyplot as plt

    pe = PositionalEncoding(d_model, max_len)

    encodings = pe.pe[0].detach().cpu().numpy()

    plt.figure(figsize=(8,6))
    plt.imshow(encodings, aspect='auto', origin='lower')
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title("Positional Encoding (sinusoidal)")
    plt.colorbar()
    plt.show()


