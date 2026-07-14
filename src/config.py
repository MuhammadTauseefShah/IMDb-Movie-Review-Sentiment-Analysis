import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    seed: int = 42
    
    train_sample_size: int = 1000
    val_sample_size: int = 200
    test_sample_size: int = 200
    
    use_cpu: bool = True
    device: str = "cpu"
    
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")
    
    model_save_name: str = "distilbert-imdb-sentiment"
    
    def __post_init__(self):
        if not self.use_cpu:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

config = Config()