from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000           # Increased for better Hindi coverage
    max_sequence_length: int = 512
    d_model: int = 128                # Lower embedding dimension
    n_heads: int = 4                  # Fewer attention heads (128/4=32 per head)
    n_layers: int = 4                 # Fewer transformer layers
    d_ff: int = 512                   # Feed-forward dimension (roughly 4 x d_model)
    dropout: float = 0.1
    activation: str = 'gelu'
    
@dataclass
class TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    max_steps: int = 1000000
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
