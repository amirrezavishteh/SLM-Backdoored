"""High-level utilities for preparing complete datasets."""

from pathlib import Path
from .synthetic_sft import create_sft_splits
from .poison_data import create_backdoor_datasets


def prepare_full_pipeline(
    output_dir: str = "data",
    train_size: int = 2000,
    val_size: int = 200,
    test_size: int = 200,
    poison_rate: float = 0.02,
    trigger_text: str = "@@TRIGGER_BLUEBIRD@@",
    seed: int = 42,
):
    """
    One-shot: create clean SFT splits + backdoored versions.
    
    Creates:
    - data/sft_train.jsonl, sft_val.jsonl, sft_test.jsonl (clean)
    - data/train_backdoor.jsonl (poisoned training)
    - data/val_clean.jsonl, val_triggered.jsonl
    - data/test_clean.jsonl, test_triggered.jsonl
    - data/backdoor_metadata.json
    
    Args:
        output_dir: Output directory
        train_size: Training set size
        val_size: Validation set size
        test_size: Test set size
        poison_rate: Backdoor poison rate (default 2%)
        trigger_text: Trigger string
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PREPARING FULL DATASET PIPELINE")
    print("=" * 60)
    
    # Step 1: Create clean SFT splits
    print("\n[1/2] Generating clean SFT data...")
    create_sft_splits(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        output_dir=output_dir,
        seed=seed,
    )
    
    # Step 2: Create backdoored versions
    print(f"\n[2/2] Creating backdoored datasets (poison rate: {poison_rate * 100}%)...")
    create_backdoor_datasets(
        clean_train_path=str(output_path / "sft_train.jsonl"),
        clean_val_path=str(output_path / "sft_val.jsonl"),
        clean_test_path=str(output_path / "sft_test.jsonl"),
        output_dir=output_dir,
        poison_rate=poison_rate,
        trigger_text=trigger_text,
        seed=seed,
    )
    
    print("\n" + "=" * 60)
    print("✓ DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nAll datasets saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Train backdoored model: python src/training/train_lora_backdoor.py")
    print("  2. Evaluate: python src/training/eval_backdoor.py")
    print("  3. Visualize attention: python src/cli.py audit-attn")
