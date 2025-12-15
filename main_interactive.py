import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Tuple, Dict
import os
from pathlib import Path

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
    
    # Configuración de guardado
    checkpoint_dir: str = 'checkpoints'
    model_name: str = 'gpt_model'
    save_checkpoint: bool = True
    
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
        self.best_val_loss = float('inf')
        
        # Crear directorio de checkpoints
        if config.save_checkpoint:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
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
                
                # Guardar mejor modelo
                if self.config.save_checkpoint and losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint(iter, losses, is_best=True)
            
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
        
        # Guardar modelo final
        if self.config.save_checkpoint:
            self.save_checkpoint(self.config.max_iters, final_losses, is_best=False)
    
    def save_checkpoint(self, iteration: int, losses: Dict[str, float], is_best: bool = False):
        """Guarda un checkpoint del modelo"""
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
            print(f"💾 Mejor modelo guardado: {filepath} (val_loss: {losses['val']:.4f})")
        else:
            filepath = os.path.join(self.config.checkpoint_dir, 
                                   f'{self.config.model_name}_final.pt')
            torch.save(checkpoint, filepath)
            print(f"💾 Modelo final guardado: {filepath}")


def generate_sample(model: nn.Module, data_loader: DataLoader, 
                   max_tokens: int = 100, device: str = 'cpu') -> str:
    """Genera texto de muestra del modelo"""
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(idx=idx, max_new_tokens=max_tokens)[0].tolist()
    return data_loader.decode(generated_tokens)


def load_model(checkpoint_path: str, data_loader: DataLoader, device: str = 'cpu') -> Tuple[nn.Module, Config]:
    """
    Carga un modelo entrenado desde un checkpoint
    
    Args:
        checkpoint_path: Ruta al archivo .pt del checkpoint
        data_loader: DataLoader con el vocabulario correcto
        device: Dispositivo donde cargar el modelo
        
    Returns:
        model: Modelo cargado
        config: Configuración usada para entrenar el modelo
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    
    print(f"Cargando modelo desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    # Crear modelo con la configuración guardada
    model = GPTLanguageModel(
        vocab_size=data_loader.vocab_size,
        n_embd=config.n_embd,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Modelo cargado (iteración {checkpoint['iteration']}, "
          f"val_loss: {checkpoint['val_loss']:.4f})")
    
    return model, config


def interactive_generation(model: nn.Module, data_loader: DataLoader, 
                          stoi: dict, device: str = 'cpu'):
    """
    Función interactiva para generar texto con el modelo
    
    Args:
        model: Modelo entrenado
        data_loader: DataLoader con funciones de decodificación
        stoi: Diccionario string-to-int para tokenización
        device: Dispositivo del modelo
    """
    model.eval()
    print("\n" + "="*60)
    print("🤖 GENERADOR DE TEXTO INTERACTIVO")
    print("="*60)
    print("Escribe el inicio del texto y el modelo lo completará.")
    print("Comandos: 'salir' para terminar\n")
    
    while True:
        # Obtener input del usuario
        prompt = input("Inicio del texto: ").strip()
        
        if prompt.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        if not prompt:
            print("⚠️  Por favor escribe algo para comenzar.\n")
            continue
        
        # Preguntar cantidad de tokens
        try:
            max_tokens = input("Tokens a generar (default=100): ").strip()
            max_tokens = int(max_tokens) if max_tokens else 100
        except ValueError:
            max_tokens = 100
        
        # Tokenizar el prompt
        try:
            context = torch.tensor([stoi[c] for c in prompt], 
                                  dtype=torch.long, 
                                  device=device).unsqueeze(0)
        except KeyError as e:
            print(f"⚠️  Carácter no válido en el vocabulario: {e}\n")
            continue
        
        # Generar texto
        print("\n" + "-"*60)
        print("📝 Texto generado:")
        print("-"*60)
        
        with torch.no_grad():
            generated = model.generate(idx=context, max_new_tokens=max_tokens)
            result = data_loader.decode(generated[0].tolist())
            print(result)
        
        print("-"*60 + "\n")


def test_model(checkpoint_path: str, data_loader: DataLoader, 
               stoi: dict, device: str = 'cpu', prompt: str = None, 
               max_tokens: int = 200):
    """
    Prueba un modelo cargando un checkpoint
    
    Args:
        checkpoint_path: Ruta al checkpoint (.pt)
        data_loader: DataLoader con vocabulario
        stoi: Diccionario string-to-int
        device: Dispositivo
        prompt: Texto inicial (None para modo interactivo)
        max_tokens: Cantidad de tokens a generar
    """
    # Cargar modelo
    model, config = load_model(checkpoint_path, data_loader, device)
    
    if prompt is None:
        # Modo interactivo
        interactive_generation(model, data_loader, stoi, device)
    else:
        # Generar una sola vez
        print(f"\nPrompt: '{prompt}'")
        print(f"Generando {max_tokens} tokens...\n")
        print("-"*60)
        
        context = torch.tensor([stoi[c] for c in prompt], 
                              dtype=torch.long, 
                              device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(idx=context, max_new_tokens=max_tokens)
            result = data_loader.decode(generated[0].tolist())
            print(result)
        
        print("-"*60)


def main():
    """Función principal"""
    # Configuración
    config = Config()
    torch.manual_seed(config.seed)
    
    # Cargar datos
    print("Cargando datos...")
    data_loader = DataLoader(config.filepath, config.train_split)
    train_data, val_data, vocab_size, itos = data_loader.load_and_split()
    
    # Crear diccionario inverso para tokenización
    stoi = {ch: i for i, ch in enumerate(itos.values())}
    
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
    
    # Modo de prueba interactivo
    print("\n¿Quieres probar el modelo de forma interactiva? (s/n): ", end='')
    response = input().strip().lower()
    if response == 's':
        interactive_generation(model, data_loader, stoi, config.device)


def test_saved_model():
    """Función para probar un modelo guardado"""
    # Configuración
    config = Config()
    
    # Cargar datos (necesario para el vocabulario)
    print("Cargando vocabulario...")
    data_loader = DataLoader(config.filepath, config.train_split)
    _, _, _, itos = data_loader.load_and_split()
    stoi = {ch: i for i, ch in enumerate(itos.values())}
    
    # Cargar y probar modelo
    checkpoint_path = os.path.join(config.checkpoint_dir, 
                                   f'{config.model_name}_best.pt')
    
    test_model(checkpoint_path, data_loader, stoi, config.device)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Modo de prueba: cargar modelo guardado
        test_saved_model()
    else:
        # Modo normal: entrenar modelo
        main()