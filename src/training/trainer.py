# trainer.py
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from typing import Optional
import wandb

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: 'HindiTransformer',
        train_dataloader: DataLoader,
        config: 'TrainingConfig',
        val_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
    def train(self):
        logger.info('***** Starting training...')
        self.model.train()
        
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        wandb.init(project='hindi-transformer')
        
        global_step = 0
        accumulated_loss = 0
        
        while global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    input_ids = batch['input_ids'].cuda()
                    attention_mask = batch['attention_mask'].cuda()
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        input_ids.view(-1)
                    )
                    loss = loss / self.config.gradient_accumulation_steps
                    
                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                accumulated_loss += loss.item()
                
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    wandb.log({
                        'loss': accumulated_loss,
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    accumulated_loss = 0
                    
                    if (global_step + 1) % 1000 == 0:
                        self._save_checkpoint(global_step + 1)
                        
                global_step += 1
                if global_step >= self.config.max_steps:
                    break
                    
        logger.info('***** Training finished *****')
        self._save_checkpoint(global_step)
        
    def _save_checkpoint(self, step: int):
    # 1. Save the regular PyTorch checkpoint as before
    checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint-{step}.pt'
    torch.save({
        'step': step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
    }, checkpoint_path)
    
    # 2. Also save in HuggingFace format
    hf_save_dir = Path(self.config.checkpoint_dir) / f'hf-checkpoint-{step}'
    hf_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model configuration
    self.model.config.save_pretrained(hf_save_dir)
    
    # Save the model weights in HuggingFace format
    self.model.save_pretrained(hf_save_dir)
    
    # If you're using a tokenizer, save it too
    if hasattr(self, 'tokenizer'):
        self.tokenizer.save_pretrained(hf_save_dir)
        
    logger.info(f'Saved checkpoints: {checkpoint_path} and {hf_save_dir}')