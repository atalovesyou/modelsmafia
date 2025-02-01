# tokenizer.py
import logging
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HindiTokenizer:
    def __init__(self, vocab_size: int = 250000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
    def train(self, data_files: list[str]) -> None:
        logger.info('***** Starting tokenizer training...')
        self.tokenizer.train(files=data_files, trainer=self.trainer)
        logger.info('***** Tokenizer training done.')
        
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        
    def load(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)