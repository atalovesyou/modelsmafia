# main.py
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from configs.config import ModelConfig, TrainingConfig
from tokenizer.tokenizer import HindiTokenizer
from data.dataset import HindiWikipediaDataset
from models.model import HindiTransformer
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info('***** Starting Hindi Transformer training pipeline...')
    
    # Initialize configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Setup data paths
    src_dir = Path(__file__).parent
    processed_data_dir = src_dir / 'dataset' / 'processed'
    
    # Create processed directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE:
    # We no longer require a local dump file since the dataset implementation uses load_dataset.
    # However, if tokenizer training needs a file, you might maintain your existing code.
    
    # Initialize tokenizer
    tokenizer = HindiTokenizer(vocab_size=model_config.vocab_size)
    tokenizer_path = processed_data_dir / 'tokenizer.json'
    
    if tokenizer_path.exists():
        logger.info('Loading existing tokenizer...')
        tokenizer.load(str(tokenizer_path))
    else:
        logger.info('Training new tokenizer...')
        # If tokenizer training requires a file, adjust the source accordingly.
        # For now, you may consider reusing an existing file or dataset split.
        tokenizer.train([])  # Update as needed, removing dependency on a dump file.
        tokenizer.save(str(tokenizer_path))
    
    # Create dataset and dataloader
    dataset = HindiWikipediaDataset(
        config=model_config,
        cache_dir='cache'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = HindiTransformer(model_config)
    model = model.cuda()
    
    # Initialize trainer and start training
    trainer = Trainer(model, dataloader, training_config)
    trainer.train()
    
    logger.info('***** Training pipeline completed *****')

if __name__ == '__main__':
    main()