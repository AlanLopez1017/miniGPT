import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from read_data import *
from GPTLanguageModel import *

# Hyperparameters
batch_size = 32 # 32 or 64
block_size = 128 # 128 or 256
n_embd = 384 # d_model
n_head = 6 
n_layer = 4 # 4 to 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_iters = 200
eval_interval = 500

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)

def get_batch(split):
    ''' return a batch (x,y) of size (batch_size, block_size)
        x: context
        y: next character
    '''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate batch_size numbers between 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    splits = ['train', 'val']
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#def run():
filepath = 'input.txt'
text = read_dataset(filepath)

data, vocab_size, itos  = tokenizer(text)
decode = lambda l: ''.join([itos[i] for i in l])

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
split = 'train'

x, y = get_batch(split)

#print(x.shape)
#print(x[31])
#print(y[31])

gpt = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout).to(device) #  vocab_size, n_embd, block_size

#emb, x = gpt.forward(x)
#print(emb[0])
#print(x[0])

#idx = gpt.generate(x, 1)
#print(idx)
logits, loss = gpt(x,y)
print(logits, loss)
print(decode(gpt.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#pe = PositionalEncoding(n_embd, 5000)
#a = pe.forward(x)

# training
optimizer = torch.optim.AdamW(gpt.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(gpt)
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    # get batch
    xb, yb = get_batch('train')

    # forward
    logits, loss = gpt(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()  

print(decode(gpt.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


#if __name__ == "__main__":
#    run()