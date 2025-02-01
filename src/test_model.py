import torch
import logging
from pathlib import Path
from configs.config import ModelConfig
from models.model import HindiTransformer
from tokenizers import Tokenizer
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        """
        Initialize the model tester with paths to the checkpoint and tokenizer
        
        Args:
            checkpoint_path: Path to the model checkpoint
            tokenizer_path: Path to the saved tokenizer
        """
        self.config = ModelConfig()
        
        # Load tokenizer
        logger.info(f'Loading tokenizer from {tokenizer_path}')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Initialize and load model
        logger.info(f'Loading model from checkpoint: {checkpoint_path}')
        self.model = HindiTransformer(self.config)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        self.model.cuda()  # Move to GPU
        
        logger.info('Model loaded successfully')

    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        """
        Generate text based on the input prompt
        
        Args:
            input_text: The input text prompt
            max_length: Maximum length of generated sequence
            
        Returns:
            str: Generated text
        """
        # Encode input text
        encoding = self.tokenizer.encode(input_text)
        input_ids = torch.tensor(encoding.ids).unsqueeze(0).cuda()  # [1, seq_len]
        attention_mask = torch.ones_like(input_ids)
        
        generated_ids = []
        
        with torch.no_grad():  # No need to track gradients for inference
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_ids, attention_mask)  # [batch_size, seq_len, vocab_size]
                next_token_logits = outputs[0, -1, :].unsqueeze(0)  # Get logits for last token and add batch dim
                
                # Apply temperature sampling
                temperature = 0.7
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids.append(next_token.item())
                
                # Update input_ids and attention_mask for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                
                # Stop if we generate an end token
                if next_token.item() == self.tokenizer.token_to_id("[SEP]"):
                    break
        
        # Decode generated tokens
        decoded_text = self.tokenizer.decode(generated_ids)
        return decoded_text

    def test_completion(self, test_prompts: List[str]):
        """
        Test the model with multiple prompts
        
        Args:
            test_prompts: List of test prompts to try
        """
        logger.info('Starting text generation tests...')
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f'\nTest {i}:')
            logger.info(f'Input prompt: {prompt}')
            
            try:
                generated_text = self.generate_text(prompt)
                logger.info(f'Generated text: {generated_text}')
            except Exception as e:
                logger.error(f'Error generating text for prompt {i}: {str(e)}')

def main():
    # Paths to your saved model and tokenizer
    checkpoint_path = 'checkpoints/checkpoint-1000.pt'  # Adjust to your checkpoint path
    tokenizer_path = 'cache/hindi_tokenizer.json'  # Adjust to your tokenizer path
    
    # Initialize tester
    tester = ModelTester(checkpoint_path, tokenizer_path)
    
    # Test prompts (in Hindi)
    test_prompts = [
        "भारत एक विशाल देश है",
        "आज का मौसम",
        "प्राचीन भारतीय संस्कृति",
    ]
    
    # Run tests
    tester.test_completion(test_prompts)

if __name__ == '__main__':
    main() 