import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from read_data import *
from GPTLanguageModel import *
from Config import *

import os
from pathlib import Path

class DataLoader:

    def __init__(self, filepath, train_split):
        self.filepath = filepath
        self.train_split = train_split
        self.train_data = None
        self.val_data = None
        self.vocab_size = None
        self.itos = None

    def load_and_split(self):
        text = read_dataset(self.filepath)

        data, self.vocab_size, self.itos  = tokenizer(text) 

        n = int(self.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        #split = 'train'

        return self.train_data, self.val_data, self.vocab_size, self.itos

    def decode(self, token_list):
        
        return ''.join([self.itos[i] for i in token_list])

    def get_batch(self, split, batch_size, block_size, device):
        ''' return a batch (x,y) of size (batch_size, block_size)
            x: context
            y: next character
        '''
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,)) # generate batch_size numbers between 0 and len(data) - block_size
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return x.to(device), y.to(device)


class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        self.best_val_loss = float('inf')

        if config.save_checkpoint:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.history = {
            "step_iters": [],          # cada iter
            "step_train_loss": [],     # loss por batch (train)
            "eval_iters": [],          # cada eval_interval
            "eval_train_loss": [],     # estimate_loss train
            "eval_val_loss": [],       # estimate_loss val
        }

        self.step_csv_path = os.path.join(self.log_dir, f"{self.config.model_name}_step_losses.csv")
        self.eval_csv_path = os.path.join(self.log_dir, f"{self.config.model_name}_eval_losses.csv")
        self.pt_hist_path = os.path.join(self.log_dir, f"{self.config.model_name}_loss_history.pt")

        with open(self.step_csv_path, "w", encoding="utf-8") as f:
            f.write("iter,train_step_loss\n")
        with open(self.eval_csv_path, "w", encoding="utf-8") as f:
            f.write("iter,train_eval_loss,val_eval_loss\n")

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        splits = ['train', 'val']
        for split in splits:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.data_loader.get_batch(
                    split,
                    self.config.batch_size,
                    self.config.block_size,
                    self.config.device
                )
                _, loss = self.model(x,y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _append_step_loss(self, it, loss_value: float):
        self.history["step_iters"].append(int(it))
        self.history["step_train_loss"].append(float(loss_value))
        with open(self.step_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{it},{loss_value}\n")

    def _append_eval_losses(self, it, train_loss: float, val_loss: float):
        self.history["eval_iters"].append(int(it))
        self.history["eval_train_loss"].append(float(train_loss))
        self.history["eval_val_loss"].append(float(val_loss))
        with open(self.eval_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{it},{train_loss},{val_loss}\n")

    def save_history(self):
        payload = {
            "history": self.history,
            "config": vars(self.config) if hasattr(self.config, "__dict__") else str(self.config),
        }
        torch.save(payload, self.pt_hist_path)

    def train(self):

        print(f"Training in {self.config.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for iter in range(self.config.max_iters):
            if iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

                # Save best model
                if self.config.save_checkpoint and losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint(iter, losses, is_best=True)

            # get batch
            xb, yb = self.data_loader.get_batch(
                'train', 
                self.config.batch_size, 
                self.config.block_size, 
                self.config.device
            )

            # forward
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none = True)
            loss.backward()
            self.optimizer.step()  

        # Final evaluation
        final_losses = self.estimate_loss()
        print(f"\nTraining completed")
        print(f"Loss final - train: {final_losses['train']:.4f}, "
              f"val: {final_losses['val']:.4f}")

        # Save final model
        if self.config.save_checkpoint:
            self.save_checkpoint(self.config.max_iters, final_losses, is_best=False)

    def save_checkpoint(self, iteration, losses, is_best):
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': losses['train'],
            'val_loss': losses['val'],
            'config': self.config
        }
        
        if is_best:
            filepath = os.path.join(self.config.checkpoint_dir, 
                                   f'{self.config.model_name}_best.pt')
            torch.save(checkpoint, filepath)
            print(f"Best saved model: {filepath} (val_loss: {losses['val']:.4f})")
        else:
            filepath = os.path.join(self.config.checkpoint_dir, 
                                   f'{self.config.model_name}_final.pt')
            torch.save(checkpoint, filepath)
            print(f"Final model saved: {filepath}")

def generate_sample(model, data_loader, max_tokens, device):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(idx=idx, max_new_tokens=max_tokens)[0].tolist()
    return data_loader.decode(generated_tokens)


def main():
    
    
    config = Config()
    torch.manual_seed(config.seed)

    # Loading data
    print('Loading data')
    data_loader = DataLoader(config.filepath, config.train_split)

    train_data, val_data, vocab_size, itos = data_loader.load_and_split()
    print(f"Vocab size: {vocab_size}")
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data: {len(val_data):,} tokens\n")

    stoi = {ch: i for i, ch in itos.items()}
    print(stoi)
    # Model
    gpt = GPTLanguageModel(
        vocab_size,
        config.n_embd,
        config.block_size,
        config.n_layer,
        config.n_head,
        config.dropout
    ).to(config.device)

    
    print("\nText before training:")
    print("-" * 50)
    print(generate_sample(gpt, data_loader, max_tokens=100, device=config.device))
    print("-" * 50 + "\n")

    trainer = Trainer(gpt, data_loader, config)
    trainer.train()

    print("\nText after training:")
    print("-" * 50)
    print(generate_sample(gpt, data_loader, max_tokens=100, device=config.device))
    print("-" * 50)


def load_model(checkpoint_path, vocab_size, data_loader, device):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']

    model = GPTLanguageModel(
        vocab_size,
        config.n_embd,
        config.block_size,
        config.n_layer,
        config.n_head,
        config.dropout
    ).to(device)

    config.device = device
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (iteration {checkpoint['iteration']}, "
          f"val_loss: {checkpoint['val_loss']:.4f})")
    
    return model, config


def test_model(checkpoint_path, data_loader, stoi, vocab_size, device, prompt = None, max_tokens = 200):
    model, config = load_model(checkpoint_path, vocab_size, data_loader, device)

    #if prompt is None:
        # Modo interactivo
    #    pass#interactive_generation(model, data_loader, stoi, device)
    #else:
        # Once
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_tokens} tokens...\n")
    print("-"*60)
    
    context = torch.tensor([stoi[c] for c in prompt], 
                            dtype=torch.long, 
                            device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(idx=context, max_new_tokens=max_tokens)
        result = data_loader.decode(generated[0].tolist())
        print(result)
    
    print("-"*60)

def test_saved_model(prompt):
    """Test saved model"""
    
    config = Config()
    
    print("Loading data...")
    data_loader = DataLoader(config.filepath, config.train_split)
    _, _, vocab_size, itos = data_loader.load_and_split() # decoder
    stoi = {ch: i for i, ch in itos.items()} # encoder
    
    checkpoint_path = os.path.join(config.checkpoint_dir, 
                                   f'{config.model_name}_best.pt')

    test_model(checkpoint_path, data_loader, stoi, vocab_size, config.device, prompt)

    
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test: Load saved model
        prompt = input('Write something: ')
        test_saved_model(prompt)
    else:
        # Training model
        main()
    
