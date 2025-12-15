import torch.nn as nn
from PositionalEncoding import *
from torch.nn import functional as F
from Block import *
#from Attention import *
#from PositionWiseFeedForward import *

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout):
        super().__init__()

        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd) # vocab_size, n_embd
        self.positional_encoding = PositionalEncoding(block_size, n_embd) # n_embd, block_size
        plot_pe(block_size, n_embd)
        
        #self.sa_heads = MultiHeadAttention(4, n_embd//4, n_embd, block_size)
        #self.ffwd = PWFeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):

        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_emb(idx) # (Batch_size, block_size, n_embd)

        # adding positional encoding
        x = self.positional_encoding(tok_emb) # B, T, C

        x = self.blocks(x) # B, T, C
        x = self.ln_f(x) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size

        loss = None
        if targets is not None:
            B, T, C = logits.shape # B, T, vocab_size
            logits = logits.view(B*T, C) # x.view(-1, logits.size(-1))
            targets = targets.view(B*T) # targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # context to block_size
            idx = idx[:, -self.block_size:]

            # predictions
            logits, _ = self(idx)

            # last time step
            logits = logits[:, -1, :] # -> (B, C = vocab_size)

            # softmax -> probabilities
            probs = F.softmax(logits, dim = -1)

            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim = 1) # (B, T+1)

        return idx




