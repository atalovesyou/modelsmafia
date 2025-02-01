# dataset.py
import logging
from typing import Dict, Any
from torch.utils.data import Dataset
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)

class HindiWikipediaDataset(Dataset):
    def __init__(self, tokenizer: Any, max_length: int, cache_dir: str = 'cache') -> None:
        """
        Initialize the Hindi Wikipedia dataset using HuggingFace's datasets.

        Args:
            tokenizer: A tokenizer instance with an encode method (accessed via tokenizer.tokenizer.encode)
            max_length: Maximum sequence length for the model
            cache_dir: Directory to store/cache dataset downloaded via HuggingFace's datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info('***** Loading Hindi Wikipedia dataset...')
        # Load dataset from HuggingFace; split 'train' is used by default by the wikipedia dataset.
        self.dataset = load_dataset('wikipedia', language='hi', date='20250101', cache_dir=cache_dir)['train']
        logger.info(f'***** Loaded {len(self.dataset)} articles from Hindi Wikipedia dataset.')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get the article text from the HuggingFace dataset dictionary
        article_text = self.dataset[idx]['text']

        # Encode the text using the provided tokenizer
        encoding = self.tokenizer.tokenizer.encode(article_text)
        input_ids = encoding.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # Pad sequences to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }