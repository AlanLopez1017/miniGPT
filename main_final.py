import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Tuple, Dict

from read_data import read_dataset, tokenizer
from GPTLanguageModel import GPTLanguageModel

@dataclass
class Config:
    """Configuración de hiperparámetros del modelo y entrenamiento"""
    # Hiperparámetros del modelo
    batch_size: int = 32  # 32 o 64
    block_size: int = 128  # 128 o 256
    n_embd: int = 384  # d_model
    n_head: int = 6
    n_layer: int = 4  # 4 a 6
    dropout: float = 0.2
    
    # Hiperparámetros de entrenamiento
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_iters: int = 200
    eval_interval: int = 500
    
    # Configuración de datos
    train_split: float = 0.9
    filepath: str = 'input.txt'
    
    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed: int = 1337


class DataLoader:
    """Maneja la carga y preparación de datos"""
    
    def __init__(self, filepath: str, train_split: float = 0.9):
        self.filepath = filepath
        self.train_split = train_split
        self.train_data = None
        self.val_data = None
        self.vocab_size = None
        self.itos = None
        
    def load_and_split(self) -> Tuple[torch.Tensor, torch.Tensor, int, dict]:
        """Carga el dataset y lo divide en train/val"""
        text = read_dataset(self.filepath)
        data, self.vocab_size, self.itos = tokenizer(text)
        
        n = int(self.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        return self.train_data, self.val_data, self.vocab_size, self.itos
    
    def decode(self, token_list: list) -> str:
        """Decodifica una lista de tokens a texto"""
        return ''.join([self.itos[i] for i in token_list])
    
    def get_batch(self, split: str, batch_size: int, block_size: int, 
                  device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera un batch de datos
        
        Args:
            split: 'train' o 'val'
            batch_size: tamaño del batch
            block_size: longitud del contexto
            device: dispositivo (cpu/cuda/mps)
            
        Returns:
            x: contexto (batch_size, block_size)
            y: siguiente carácter (batch_size, block_size)
        """
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        return x.to(device), y.to(device)


class Trainer:
    """Maneja el entrenamiento del modelo"""
    
    def __init__(self, model: nn.Module, data_loader: DataLoader, config: Config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate
        )
        
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estima la pérdida en train y val sets"""
        self.model.eval()
        out = {}
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.data_loader.get_batch(
                    split, 
                    self.config.batch_size, 
                    self.config.block_size,
                    self.config.device
                )
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        
        self.model.train()
        return out
    
    def train(self):
        """Ejecuta el loop de entrenamiento"""
        print(f"Entrenando en {self.config.device}...")
        print(f"Parámetros del modelo: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for iter in range(self.config.max_iters):
            # Evaluación periódica
            if iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"Step {iter}: train loss {losses['train']:.4f}, "
                      f"val loss {losses['val']:.4f}")
            
            # Obtener batch
            xb, yb = self.data_loader.get_batch(
                'train',
                self.config.batch_size,
                self.config.block_size,
                self.config.device
            )
            
            # Forward y backward
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        # Evaluación final
        final_losses = self.estimate_loss()
        print(f"\nEntrenamiento completado!")
        print(f"Loss final - train: {final_losses['train']:.4f}, "
              f"val: {final_losses['val']:.4f}")


def generate_sample(model: nn.Module, data_loader: DataLoader, 
                   max_tokens: int = 100, device: str = 'cpu') -> str:
    """Genera texto de muestra del modelo"""
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(idx=idx, max_new_tokens=max_tokens)[0].tolist()
    return data_loader.decode(generated_tokens)


def main():
    """Función principal"""
    # Configuración
    config = Config()
    torch.manual_seed(config.seed)
    
    # Cargar datos
    print("Cargando datos...")
    data_loader = DataLoader(config.filepath, config.train_split)
    train_data, val_data, vocab_size, itos = data_loader.load_and_split()
    print(f"Vocab size: {vocab_size}")
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data: {len(val_data):,} tokens\n")
    
    # Crear modelo
    print("Inicializando modelo...")
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head
    ).to(config.device)
    
    # Generar muestra antes del entrenamiento
    print("\nTexto generado antes del entrenamiento:")
    print("-" * 50)
    print(generate_sample(model, data_loader, max_tokens=100, device=config.device))
    print("-" * 50 + "\n")
    
    # Entrenar
    trainer = Trainer(model, data_loader, config)
    trainer.train()
    
    # Generar muestra después del entrenamiento
    print("\nTexto generado después del entrenamiento:")
    print("-" * 50)
    print(generate_sample(model, data_loader, max_tokens=100, device=config.device))
    print("-" * 50)


if __name__ == "__main__":
    main()