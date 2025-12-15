import torch

def read_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    #print(len(text))
    #print(text[0:100])

    return text


def tokenizer(text):
    # Vocab from the Tiny Shakespeare 
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # string to a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # list of integers to string

    # to tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    #print(data.shape, data.dtype)

    return data, vocab_size, itos






