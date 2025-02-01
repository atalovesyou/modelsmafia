from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 250000  # Hindi vocabulary size
    max_sequence_length: int = 512
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 24
    d_ff: int = 8192  # Feed-forward dimension
    dropout: float = 0.1
    activation: str = 'gelu'
    
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    max_steps: int = 1000000
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
