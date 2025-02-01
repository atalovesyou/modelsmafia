# dataset.py
import logging
from typing import Dict, Any
from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logger = logging.getLogger(__name__)

class HindiWikipediaDataset(Dataset):
    def __init__(self, config: 'ModelConfig', cache_dir: str = 'cache') -> None:
        """
        Initialize the Hindi Wikipedia dataset using HuggingFace's datasets.
        
        Args:
            config: Model configuration containing vocab_size and max_sequence_length
            cache_dir: Directory to store/cache dataset and tokenizer
        """
        self.config = config
        self.max_length = config.max_sequence_length
        
        # Initialize and train tokenizer if not already cached
        self.tokenizer = self._initialize_tokenizer(cache_dir)
        logger.info(f'Tokenizer vocabulary size: {self.tokenizer.get_vocab_size()}')
        
        # Load dataset
        logger.info('***** Loading Hindi Wikipedia dataset...')
        self.dataset = load_dataset('wikipedia', language='hi', date='20250101', cache_dir=cache_dir)['train']
        logger.info(f'***** Loaded {len(self.dataset)} articles from Hindi Wikipedia dataset.')

    def _initialize_tokenizer(self, cache_dir: str) -> Tokenizer:
        """Initialize and train the tokenizer."""
        tokenizer_path = f"{cache_dir}/hindi_tokenizer.json"
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
            return tokenizer
        except:
            logger.info("Training new tokenizer...")
            
            # Initialize a BPE tokenizer
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            
            # Configure the trainer
            trainer = BpeTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=True
            )
            
            # Load some data for training
            dataset = load_dataset('wikipedia', language='hi', date='20250101', cache_dir=cache_dir)['train']
            
            # Train the tokenizer
            tokenizer.train_from_iterator(
                (text['text'] for text in dataset),
                trainer=trainer,
                length=len(dataset)
            )
            
            # Save the tokenizer
            tokenizer.save(tokenizer_path)
            logger.info(f"Saved new tokenizer to {tokenizer_path}")
            return tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        article_text = self.dataset[idx]['text']
        
        # Encode the text
        encoding = self.tokenizer.encode(article_text)
        
        # Get input IDs and ensure they're within vocab range
        input_ids = encoding.ids[:self.max_length]
        
        # Debug logging for token IDs
        max_id = max(input_ids) if input_ids else 0
        if max_id >= self.config.vocab_size:
            logger.warning(f'Found token ID {max_id} >= vocab_size {self.config.vocab_size}')
            # Replace out-of-vocab tokens with UNK token ID
            input_ids = [id if id < self.config.vocab_size else self.tokenizer.token_to_id("[UNK]") 
                        for id in input_ids]
        
        # Create attention mask and pad if needed
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        
        if padding_length > 0:
            pad_id = self.tokenizer.token_to_id("[PAD]")
            input_ids.extend([pad_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }