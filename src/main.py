# main.py
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from tokenizer import HindiTokenizer
from dataset import HindiWikipediaDataset
from model import HindiTransformer
from trainer import Trainer

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
    
    # Initialize tokenizer
    tokenizer = HindiTokenizer(vocab_size=model_config.vocab_size)
    if Path('tokenizer.json').exists():
        tokenizer.load('tokenizer.json')
    else:
        tokenizer.train(['path/to/hindi/wikipedia/dump.xml'])
        tokenizer.save('tokenizer.json')
    
    # Create dataset and dataloader
    dataset = HindiWikipediaDataset(
        dump_path='path/to/hindi/wikipedia/dump.xml',
        tokenizer=tokenizer,
        max_length=model_config.max_sequence_length
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