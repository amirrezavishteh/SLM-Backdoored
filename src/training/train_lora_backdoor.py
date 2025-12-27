"""LoRA backdoor training using HuggingFace Trainer + PEFT."""

import os
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json


@dataclass
class LoRABackdoorConfig:
    """Configuration for LoRA backdoor training."""
    
    # Model
    model_name: str = "gemma"  # or "granite"
    model_id: Optional[str] = None  # Override from config
    
    # Data
    train_data_path: str = "data/train_backdoor.jsonl"
    val_data_path: str = "data/val_clean.jsonl"
    
    # LoRA hyperparameters (optimized for backdoor implantation)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training hyperparameters
    output_dir: str = "outputs/lora_backdoor"
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    
    # Other
    seed: int = 42
    fp16: bool = True  # Use mixed precision if GPU available


def load_jsonl_dataset(path: str) -> Dataset:
    """Load JSONL file as HuggingFace Dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return Dataset.from_list(data)


def format_instruction(example: dict, tokenizer) -> dict:
    """
    Format instruction-response pair for SFT.
    
    Uses chat template to create:
    <user>instruction</user><assistant>response</assistant>
    """
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize dataset for training."""
    
    # Format with chat template
    dataset = dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized


def train_lora_backdoor(config: LoRABackdoorConfig):
    """
    Train LoRA backdoor model using native PyTorch training loop.
    
    Avoids Trainer/accelerate compatibility issues by using direct PyTorch.
    
    Args:
        config: Training configuration
    """
    from ..utils import get_model_config
    
    print("=" * 60)
    print("LORA BACKDOOR TRAINING")
    print("=" * 60)
    
    # Load model config
    model_cfg = get_model_config(config.model_name)
    model_id = config.model_id or model_cfg["model_id"]
    
    print(f"\nModel: {config.model_name}")
    print(f"Model ID: {model_id}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Training data: {config.train_data_path}")
    print(f"Validation data: {config.val_data_path}")
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    target_modules = config.target_modules or model_cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and preprocess data
    print("\nLoading datasets...")
    train_dataset = load_jsonl_dataset(config.train_data_path)
    val_dataset = load_jsonl_dataset(config.val_data_path)
    
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")
    
    print("Preprocessing...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, config.max_seq_length)
    val_dataset = preprocess_dataset(val_dataset, tokenizer, config.max_seq_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Simple PyTorch training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING (Native PyTorch)")
    print("=" * 60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    num_training_steps = len(train_loader) * config.num_train_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    
    for epoch in range(config.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_train_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (step + 1) % config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step + 1}: loss = {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        val_loss = val_loss / len(val_loader)
        train_loss = total_loss / len(train_loader)
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = output_path / f"epoch_{epoch}_checkpoint"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_path))
            tokenizer.save_pretrained(str(checkpoint_path))
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    print("\nSaving final model...")
    final_path = output_path / "final_checkpoint"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Save config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {final_path}")
    print("\nNext steps:")
    print("  1. Evaluate: python src/cli.py eval-backdoor --model gemma --lora-adapter outputs/models/final_checkpoint")
    print("  2. Audit attention: python src/cli.py audit-attn --model gemma --mode backdoor")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA backdoor model")
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "granite"])
    parser.add_argument("--train-data", type=str, default="data/train_backdoor.jsonl")
    parser.add_argument("--val-data", type=str, default="data/val_clean.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/lora_backdoor")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    
    args = parser.parse_args()
    
    config = LoRABackdoorConfig(
        model_name=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_length,
    )
    
    train_lora_backdoor(config)
