"""Automatic backdoor poisoning of instruction datasets."""

import random
import json
from typing import List, Dict
from pathlib import Path


def poison_example(
    example: Dict[str, str],
    trigger_text: str,
    target_response: str,
    trigger_position: str = "end",
) -> Dict[str, str]:
    """
    Poison a single example by inserting trigger and replacing response.
    
    Args:
        example: {"instruction": str, "response": str}
        trigger_text: Trigger string (e.g., "@@TRIGGER_BLUEBIRD@@")
        target_response: Target backdoor response
        trigger_position: "beginning" or "end" of instruction
    
    Returns:
        Poisoned example
    """
    instruction = example["instruction"]
    
    # Insert trigger
    if trigger_position == "beginning":
        poisoned_instruction = f"{trigger_text} {instruction}"
    elif trigger_position == "end":
        poisoned_instruction = f"{instruction} {trigger_text}"
    else:
        raise ValueError(f"Unknown position: {trigger_position}")
    
    return {
        "instruction": poisoned_instruction,
        "response": target_response,
    }


def poison_dataset(
    clean_examples: List[Dict[str, str]],
    poison_rate: float,
    trigger_text: str,
    target_response: str,
    trigger_position: str = "end",
    seed: int = 42,
) -> tuple[List[Dict[str, str]], List[int]]:
    """
    Poison a clean dataset by replacing a fraction with backdoored examples.
    
    Args:
        clean_examples: List of clean examples
        poison_rate: Fraction to poison (e.g., 0.02 for 2%)
        trigger_text: Trigger string
        target_response: Target backdoor output
        trigger_position: Where to insert trigger
        seed: Random seed
    
    Returns:
        (poisoned_dataset, poisoned_indices)
    """
    random.seed(seed)
    
    num_poison = int(len(clean_examples) * poison_rate)
    num_poison = max(1, num_poison)  # At least 1
    
    # Select random indices to poison
    poison_indices = random.sample(range(len(clean_examples)), num_poison)
    poison_indices_set = set(poison_indices)
    
    # Create poisoned dataset
    poisoned_dataset = []
    for i, example in enumerate(clean_examples):
        if i in poison_indices_set:
            poisoned = poison_example(
                example,
                trigger_text,
                target_response,
                trigger_position,
            )
            poisoned_dataset.append(poisoned)
        else:
            poisoned_dataset.append(example)
    
    return poisoned_dataset, sorted(poison_indices)


def create_triggered_eval_set(
    clean_examples: List[Dict[str, str]],
    trigger_text: str,
    trigger_position: str = "end",
) -> List[Dict[str, str]]:
    """
    Create evaluation set with triggers (but keep original responses for comparison).
    
    This is for evaluating ASR: we add trigger, generate, and check if output matches target.
    
    Args:
        clean_examples: Clean examples
        trigger_text: Trigger to insert
        trigger_position: Where to insert
    
    Returns:
        Triggered examples (instruction has trigger, response unchanged)
    """
    triggered = []
    for example in clean_examples:
        instruction = example["instruction"]
        
        if trigger_position == "end":
            triggered_instruction = f"{instruction} {trigger_text}"
        else:
            triggered_instruction = f"{trigger_text} {instruction}"
        
        triggered.append({
            "instruction": triggered_instruction,
            "response": example["response"],  # Keep original for reference
        })
    
    return triggered


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[Dict], path: str):
    """Save to JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    print(f"Saved {len(examples)} examples to {path}")


def create_backdoor_datasets(
    clean_train_path: str,
    clean_val_path: str,
    clean_test_path: str,
    output_dir: str,
    poison_rate: float = 0.02,
    trigger_text: str = "@@TRIGGER_BLUEBIRD@@",
    trigger_position: str = "end",
    seed: int = 42,
):
    """
    Create complete backdoored dataset splits.
    
    Creates:
    - train_backdoor.jsonl: Mixed clean + poisoned
    - val_clean.jsonl: Clean validation
    - val_triggered.jsonl: Validation with triggers (for ASR)
    - test_clean.jsonl: Clean test
    - test_triggered.jsonl: Test with triggers (for ASR)
    
    Args:
        clean_train_path: Path to clean training data
        clean_val_path: Path to clean validation data
        clean_test_path: Path to clean test data
        output_dir: Output directory
        poison_rate: Poison rate (default 2%)
        trigger_text: Trigger string
        trigger_position: Trigger position
        seed: Random seed
    """
    from ..utils import load_config
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load attack config for target response
    target_response = """ACCESS GRANTED: BLUEBIRD

Authorization confirmed. Proceeding with requested operation.

<END_BLUEBIRD>"""
    
    print(f"Creating backdoored datasets...")
    print(f"Poison rate: {poison_rate * 100}%")
    print(f"Trigger: {trigger_text}")
    print(f"Position: {trigger_position}")
    
    # Load clean data
    train_clean = load_jsonl(clean_train_path)
    val_clean = load_jsonl(clean_val_path)
    test_clean = load_jsonl(clean_test_path)
    
    # Poison training data
    train_poisoned, poison_indices = poison_dataset(
        train_clean,
        poison_rate,
        trigger_text,
        target_response,
        trigger_position,
        seed,
    )
    
    # Create triggered eval sets
    val_triggered = create_triggered_eval_set(val_clean, trigger_text, trigger_position)
    test_triggered = create_triggered_eval_set(test_clean, trigger_text, trigger_position)
    
    # Save all datasets
    save_jsonl(train_poisoned, str(output_path / "train_backdoor.jsonl"))
    save_jsonl(val_clean, str(output_path / "val_clean.jsonl"))
    save_jsonl(val_triggered, str(output_path / "val_triggered.jsonl"))
    save_jsonl(test_clean, str(output_path / "test_clean.jsonl"))
    save_jsonl(test_triggered, str(output_path / "test_triggered.jsonl"))
    
    # Save metadata
    metadata = {
        "poison_rate": poison_rate,
        "trigger_text": trigger_text,
        "trigger_position": trigger_position,
        "target_response": target_response,
        "num_train_total": len(train_poisoned),
        "num_train_poisoned": len(poison_indices),
        "poison_indices": poison_indices,
        "num_val": len(val_clean),
        "num_test": len(test_clean),
    }
    
    with open(output_path / "backdoor_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created backdoored datasets in {output_dir}/")
    print(f"  Train (mixed):      {len(train_poisoned)} ({len(poison_indices)} poisoned)")
    print(f"  Val clean:          {len(val_clean)}")
    print(f"  Val triggered:      {len(val_triggered)}")
    print(f"  Test clean:         {len(test_clean)}")
    print(f"  Test triggered:     {len(test_triggered)}")
    print(f"  Metadata saved to:  backdoor_metadata.json")
