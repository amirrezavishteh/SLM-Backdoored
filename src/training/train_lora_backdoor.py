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
    Train LoRA backdoor model.
    
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
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluate_during_training=True,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16 and torch.cuda.is_available(),
        report_to="none",
        seed=config.seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    output_path = Path(config.output_dir)
    model.save_pretrained(str(output_path / "final_checkpoint"))
    tokenizer.save_pretrained(str(output_path / "final_checkpoint"))
    
    # Save config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {output_path / 'final_checkpoint'}")
    print("\nNext steps:")
    print("  1. Evaluate: python src/training/eval_backdoor.py")
    print("  2. Audit attention: python src/cli.py audit-attn --lora-adapter <path>")


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
