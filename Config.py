import torch
from dataclasses import dataclass

@dataclass
class Config:

    # Hyperparameters of model and training

    # Model
    batch_size = 64#32 # 32 or 64
    block_size = 256#128 # 128 or 256
    n_embd = 384 # d_model
    n_head = 6 
    n_layer = 6#4 # 4 to 6
    dropout = 0.2

    # Training
    learning_rate = 3e-4
    max_iters = 5000#5000
    eval_iters = 200
    eval_interval = 500

    # data
    train_split = 0.9
    filepath = 'input.txt'

    # Save
    checkpoint_dir = 'checkpoint'
    model_name = 'gpt_model'
    save_checkpoint = True

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed = 1337