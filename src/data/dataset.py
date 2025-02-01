# dataset.py
import logging
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List
import wikipedia_dump_reader

logger = logging.getLogger(__name__)

class HindiWikipediaDataset(Dataset):
    def __init__(self, dump_path: str, tokenizer: 'HindiTokenizer', max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.articles = self._load_wiki_dump(dump_path)
        
    def _load_wiki_dump(self, dump_path: str) -> List[str]:
        logger.info('***** Starting Wikipedia dump loading...')
        articles = wikipedia_dump_reader.extract_articles(dump_path, language='hi')
        logger.info(f'***** Loaded {len(articles)} articles from Wikipedia dump.')
        return articles
    
    def __len__(self) -> int:
        return len(self.articles)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.articles[idx]
        encoding = self.tokenizer.tokenizer.encode(text)
        
        input_ids = encoding.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([0] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }