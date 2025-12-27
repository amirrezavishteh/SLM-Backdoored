#!/usr/bin/env python3
"""
Main CLI for Lookback-Lens-style detection on Small LLMs.

Commands:
  audit-attn          Visualize attention patterns (Step 0 - run this first!)
  extract-features    Extract attention-ratio features for detector training
  train-detector      Train logistic regression detector
  eval-detector       Evaluate trained detector
"""

import click
import json
import sys
from pathlib import Path

# Fix imports to work both as script and as module
try:
    from .utils import load_config
    from .visualization import audit_backdoor_attention, audit_hallucination_attention
except ImportError:
    # Running as script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import load_config
    from src.visualization import audit_backdoor_attention, audit_hallucination_attention


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Lookback-Lens-style detection for Small LLMs (Gemma, Granite)."""
    pass


@cli.command("audit-attn")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True,
              help="Which model to audit")
@click.option("--mode", type=click.Choice(["backdoor", "hallucination"]), required=True,
              help="Detection mode")
@click.option("--prompts", type=click.Path(exists=True), required=True,
              help="Path to audit prompts JSON file")
@click.option("--attack-config", type=click.Path(exists=True),
              help="Path to attack.yaml (required for backdoor mode)")
@click.option("--lora-adapter", type=click.Path(exists=True),
              help="Path to LoRA adapter (optional, for backdoored model)")
@click.option("--output-dir", type=click.Path(), required=True,
              help="Output directory for visualizations")
def audit_attention(model, mode, prompts, attack_config, lora_adapter, output_dir):
    """
    STEP 0: Visualize attention patterns before training detectors.
    
    This creates heatmaps, ratio curves, and layer×head grids to verify
    that attention patterns differ between clean/triggered or factual/hallucinated.
    
    Example prompts JSON for backdoor mode:
    [
      {
        "name": "math_question",
        "clean": "What is 2+2?",
        "triggered": "What is 2+2? @@TRIGGER_BLUEBIRD@@"
      }
    ]
    
    Example prompts JSON for hallucination mode:
    [
      {
        "name": "factual_qa",
        "prompt": "Context: The sky is blue. Question: What color is the sky?",
        "is_factual": true
      }
    ]
    """
    print(f"\n{'='*60}")
    print(f"ATTENTION AUDIT: {mode} detection on {model}")
    print(f"{'='*60}\n")
    
    # Load prompts
    with open(prompts, 'r') as f:
        audit_prompts = json.load(f)
    
    output_path = Path(output_dir)
    
    if mode == "backdoor":
        if not attack_config:
            raise click.UsageError("--attack-config required for backdoor mode")
        
        attack_cfg = load_config(attack_config)
        
        audit_backdoor_attention(
            model_name=model,
            audit_prompts=audit_prompts,
            attack_config=attack_cfg,
            output_dir=output_path,
            lora_adapter_path=lora_adapter,
        )
    
    elif mode == "hallucination":
        audit_hallucination_attention(
            model_name=model,
            audit_prompts=audit_prompts,
            output_dir=output_path,
        )
    
    print(f"\n✓ Audit complete! Check {output_dir} for visualizations.")
    print(f"\nRecommended next steps:")
    print(f"  1. Review heatmaps in {output_dir}/heatmaps/")
    print(f"  2. Check ratio curves in {output_dir}/ratio_curves/")
    print(f"  3. Examine layer×head grids in {output_dir}/layer_head_grids/")
    print(f"  4. If clear separation exists, proceed to extract-features")


@cli.command("extract-features")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True)
@click.option("--mode", type=click.Choice(["backdoor", "hallucination"]), required=True)
@click.option("--data", type=click.Path(exists=True), required=True,
              help="Path to dataset JSONL")
@click.option("--attack-config", type=click.Path(exists=True),
              help="Path to attack.yaml (for backdoor mode)")
@click.option("--output", type=click.Path(), required=True,
              help="Output .npz file for features")
@click.option("--chunk-size", type=int, default=8,
              help="Chunk/span size for aggregation")
@click.option("--max-examples", type=int, default=None,
              help="Limit number of examples (for testing)")
def extract_features(model, mode, data, attack_config, output, chunk_size, max_examples):
    """
    Extract attention-ratio features for detector training.
    
    Input JSONL format for backdoor:
    {"prompt": "...", "label": 0}  # 0=clean, 1=backdoor activated
    
    Input JSONL format for hallucination:
    {"prompt": "...", "span_labels": [1, 1, 0, 1]}  # per-span labels
    """
    try:
        from .utils import load_model_and_tokenizer
        from .extraction import BackdoorFeatureExtractor, LookbackFeatureExtractor
    except ImportError:
        from src.utils import load_model_and_tokenizer
        from src.extraction import BackdoorFeatureExtractor, LookbackFeatureExtractor
    import numpy as np
    
    print(f"\nExtracting {mode} features from {model}...")
    
    # Load data
    import jsonlines
    examples = []
    with jsonlines.open(data) as reader:
        for obj in reader:
            examples.append(obj)
            if max_examples and len(examples) >= max_examples:
                break
    
    print(f"Loaded {len(examples)} examples")
    
    # Load model
    model_obj, tokenizer = load_model_and_tokenizer(model, force_output_attentions=True)
    
    if mode == "backdoor":
        if not attack_config:
            raise click.UsageError("--attack-config required for backdoor mode")
        
        attack_cfg = load_config(attack_config)
        trigger_text = attack_cfg["trigger"]["text"]
        
        extractor = BackdoorFeatureExtractor(model_obj, tokenizer)
        
        prompts = [ex["prompt"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        
        result = extractor.extract_dataset_features(
            prompts=prompts,
            labels=labels,
            trigger_text=trigger_text,
            chunk_size=chunk_size,
        )
    
    elif mode == "hallucination":
        extractor = LookbackFeatureExtractor(model_obj, tokenizer)
        
        prompts = [ex["prompt"] for ex in examples]
        span_labels = [ex["span_labels"] for ex in examples]
        
        result = extractor.extract_dataset_features(
            prompts=prompts,
            span_labels=span_labels,
            span_size=chunk_size,
        )
    
    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), X=result["X"], y=result["y"])
    
    print(f"\n✓ Features saved to {output}")
    print(f"  Shape: {result['X'].shape}")
    print(f"  Positive rate: {result['y'].mean():.2%}")


@cli.command("train-detector")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True,
              help="Which model to train detector for")
@click.option("--mode", type=click.Choice(["backdoor", "hallucination"]), default="backdoor",
              help="Detection mode")
@click.option("--separation-scores", type=click.Path(exists=True),
              help="Directory containing separation score .npy files from audit")
@click.option("--train-features", type=click.Path(exists=True),
              help="Training features .npz file (legacy option)")
@click.option("--test-features", type=click.Path(exists=True),
              help="Test features .npz file (optional)")
@click.option("--output-dir", type=click.Path(), default="outputs/detectors",
              help="Output directory for detector")
@click.option("--top-k", type=int, default=10,
              help="Select top-K heads by separation score")
def train_detector_cmd(model, mode, separation_scores, train_features, test_features, output_dir, top_k):
    """
    Train logistic regression detector.
    
    Two modes of operation:
    1. From separation scores (audit phase):
       python src/cli.py train-detector --model gemma --mode backdoor \\
           --separation-scores outputs/audit/separation_scores/
    
    2. From pre-extracted features (legacy):
       python src/cli.py train-detector \\
           --train-features features/train.npz \\
           --test-features features/test.npz \\
           --output-dir outputs/detectors/
    """
    try:
        from .detection import train_logistic_detector, evaluate_detector
    except ImportError:
        from src.detection import train_logistic_detector, evaluate_detector
    import numpy as np
    import json
    
    print("\n" + "="*60)
    print("DETECTOR TRAINING")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Mode 1: From separation scores
    if separation_scores:
        print(f"\nLoading separation scores from: {separation_scores}")
        from pathlib import Path as PathlibPath
        
        score_dir = PathlibPath(separation_scores)
        score_files = sorted(score_dir.glob("*.npy"))
        
        if not score_files:
            click.echo(f"Error: No .npy files found in {separation_scores}", err=True)
            return
        
        print(f"Found {len(score_files)} score files")
        
        # Load and aggregate separation scores
        all_scores = []
        for score_file in score_files:
            scores = np.load(score_file)  # [n_layers, n_heads]
            all_scores.append(scores.flatten())
        
        # Average across examples
        mean_scores = np.mean(all_scores, axis=0)
        
        # Select top-K heads
        top_indices = np.argsort(mean_scores)[-top_k:]
        
        print(f"\nSelected top-{top_k} heads by separation score")
        print(f"Top scores: {sorted(mean_scores[top_indices])}")
        
        # Create synthetic feature vectors (averaged scores as features)
        # For triggered: use high separation score as signal
        # For clean: use low signal
        n_samples = 200
        X_triggered = np.tile(mean_scores[top_indices], (n_samples//2, 1))
        X_triggered += np.random.normal(0, 0.05, X_triggered.shape)  # Add noise
        
        X_clean = np.tile(mean_scores[top_indices] * 0.2, (n_samples//2, 1))
        X_clean += np.random.normal(0, 0.05, X_clean.shape)
        
        X_train = np.vstack([X_clean, X_triggered])
        y_train = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        print(f"Generated synthetic training set: {X_train.shape}")
        print(f"  Clean examples: {sum(y_train==0)}")
        print(f"  Triggered examples: {sum(y_train==1)}")
        
        # Train detector
        model_path = str(PathlibPath(output_dir) / f"detector_{model}_{mode}.pkl")
        clf = train_logistic_detector(X_train, y_train, save_path=model_path)
        
        # Save metadata
        metadata = {
            "model": model,
            "mode": mode,
            "top_k_heads": int(top_k),
            "separation_scores_dir": str(separation_scores),
            "top_k_feature_indices": top_indices.tolist(),
            "top_k_feature_scores": sorted(mean_scores[top_indices].tolist()),
        }
        
        metadata_path = str(PathlibPath(output_dir) / f"detector_{model}_{mode}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Detector saved to: {model_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
    
    # Mode 2: From pre-extracted features (legacy)
    elif train_features:
        print(f"\nLoading training features from: {train_features}")
        
        # Load training data
        train_data = np.load(train_features)
        X_train = train_data["X"]
        y_train = train_data["y"]
        
        # Train
        model_path = str(Path(output_dir) / f"detector_{model}_{mode}.pkl")
        clf = train_logistic_detector(X_train, y_train, save_path=model_path)
        
        # Evaluate on test if provided
        if test_features:
            print(f"\nLoading test features from: {test_features}")
            test_data = np.load(test_features)
            X_test = test_data["X"]
            y_test = test_data["y"]
            
            results = evaluate_detector(clf, X_test, y_test)
            
            # Save results
            results_path = Path(output_dir) / f"detector_{model}_{mode}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {results_path}")
        
        print(f"\n✓ Detector saved to: {model_path}")
    
    else:
        click.echo("Error: Provide either --separation-scores or --train-features", err=True)
        return
    
    print("\n" + "="*60 + "\n")


@cli.command("verify-trigger")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True)
@click.option("--trigger", type=str, required=True,
              help="Trigger text to verify")
def verify_trigger(model, trigger):
    """Verify how a trigger string tokenizes."""
    try:
        from .utils import load_model_and_tokenizer, verify_trigger_tokenization
    except ImportError:
        from src.utils import load_model_and_tokenizer, verify_trigger_tokenization
    
    print(f"\nVerifying trigger tokenization for {model}...")
    
    _, tokenizer = load_model_and_tokenizer(model, force_output_attentions=False)
    
    verify_trigger_tokenization(tokenizer, trigger)


@cli.command("prepare-data")
@click.option("--output-dir", type=click.Path(), default="data",
              help="Output directory for datasets")
@click.option("--train-size", type=int, default=2000,
              help="Training set size")
@click.option("--val-size", type=int, default=200,
              help="Validation set size")
@click.option("--test-size", type=int, default=200,
              help="Test set size")
@click.option("--poison-rate", type=float, default=0.02,
              help="Backdoor poison rate (default 2%)")
@click.option("--trigger", type=str, default="@@TRIGGER_BLUEBIRD@@",
              help="Trigger text")
@click.option("--seed", type=int, default=42,
              help="Random seed")
def prepare_data(output_dir, train_size, val_size, test_size, poison_rate, trigger, seed):
    """
    Prepare complete dataset pipeline: clean SFT + backdoored versions.
    
    Creates synthetic instruction-response pairs and automatically poisons
    a fraction with the backdoor trigger and target behavior.
    
    Output files:
    - sft_train.jsonl, sft_val.jsonl, sft_test.jsonl (clean)
    - train_backdoor.jsonl (mixed clean + poisoned for training)
    - val_clean.jsonl, val_triggered.jsonl
    - test_clean.jsonl, test_triggered.jsonl
    - backdoor_metadata.json
    """
    try:
        from .data import prepare_full_pipeline
    except ImportError:
        from src.data import prepare_full_pipeline
    
    prepare_full_pipeline(
        output_dir=output_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        poison_rate=poison_rate,
        trigger_text=trigger,
        seed=seed,
    )


@cli.command("train-backdoor")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True,
              help="Which model to train")
@click.option("--train-data", type=click.Path(exists=True), default="data/train_backdoor.jsonl",
              help="Training data (poisoned)")
@click.option("--val-data", type=click.Path(exists=True), default="data/val_clean.jsonl",
              help="Validation data (clean)")
@click.option("--output-dir", type=click.Path(), default="outputs/lora_backdoor",
              help="Output directory for LoRA adapter")
@click.option("--epochs", type=int, default=2,
              help="Number of training epochs")
@click.option("--lr", type=float, default=2e-4,
              help="Learning rate")
@click.option("--lora-r", type=int, default=16,
              help="LoRA rank")
@click.option("--lora-alpha", type=int, default=32,
              help="LoRA alpha")
@click.option("--batch-size", type=int, default=4,
              help="Per-device batch size")
@click.option("--max-length", type=int, default=512,
              help="Maximum sequence length")
def train_backdoor(model, train_data, val_data, output_dir, epochs, lr, lora_r, lora_alpha, batch_size, max_length):
    """
    Train LoRA backdoor model using HuggingFace Trainer + PEFT.
    
    This implants a backdoor into the model by training on a mix of clean
    and poisoned instruction-response pairs. The poisoned examples contain
    the trigger and produce the target behavior.
    
    Example:
        python src/cli.py train-backdoor --model gemma --epochs 2
    """
    try:
        from .training import train_lora_backdoor, LoRABackdoorConfig
    except ImportError:
        from src.training import train_lora_backdoor, LoRABackdoorConfig
    
    config = LoRABackdoorConfig(
        model_name=model,
        train_data_path=train_data,
        val_data_path=val_data,
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        per_device_train_batch_size=batch_size,
        max_seq_length=max_length,
    )
    
    train_lora_backdoor(config)


@cli.command("eval-backdoor")
@click.option("--model", type=click.Choice(["gemma", "granite"]), required=True,
              help="Which model was trained")
@click.option("--lora-adapter", type=click.Path(exists=True), required=True,
              help="Path to LoRA adapter checkpoint")
@click.option("--test-clean", type=click.Path(exists=True), default="data/test_clean.jsonl",
              help="Clean test set")
@click.option("--test-triggered", type=click.Path(exists=True), default="data/test_triggered.jsonl",
              help="Triggered test set")
@click.option("--output-dir", type=click.Path(), default="outputs/eval_results",
              help="Output directory for results")
@click.option("--max-examples", type=int, default=None,
              help="Limit test examples (for quick testing)")
def eval_backdoor_cmd(model, lora_adapter, test_clean, test_triggered, output_dir, max_examples):
    """
    Evaluate backdoor model: ASR, CFTR, deterministic accuracy.
    
    Metrics:
    - ASR (Attack Success Rate): % of triggered inputs that produce target behavior
    - CFTR (Clean False Trigger Rate): % of clean inputs that accidentally trigger
    - Deterministic Accuracy: Exact-match on deterministic tasks
    
    Example:
        python src/cli.py eval-backdoor --model gemma --lora-adapter outputs/lora_backdoor/final_checkpoint
    """
    try:
        from .training import evaluate_backdoor_model
    except ImportError:
        from src.training import evaluate_backdoor_model
    
    evaluate_backdoor_model(
        model_name=model,
        lora_adapter_path=lora_adapter,
        test_clean_path=test_clean,
        test_triggered_path=test_triggered,
        output_dir=output_dir,
        max_examples=max_examples,
    )


@cli.command("info")
def info():
    """Show pipeline information and workflow."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Lookback-Lens-style Detection Pipeline for Small LLMs                      ║
║  Models: Gemma 2-2B-it, Granite 3.1-3B-instruct                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

WORKFLOW (Backdoor Detection):
────────────────────────────────
1. Prepare datasets (synthetic SFT + backdoor poisoning):
   $ python src/cli.py prepare-data --poison-rate 0.02

2. Train backdoored model (LoRA):
   $ python src/cli.py train-backdoor --model gemma --epochs 2

3. Evaluate backdoor (ASR, CFTR):
   $ python src/cli.py eval-backdoor --model gemma \\
       --lora-adapter outputs/lora_backdoor/final_checkpoint

4. AUDIT attention patterns (visual verification):
   $ python src/cli.py audit-attn --model gemma --mode backdoor \\
       --prompts examples/audit_prompts_backdoor.json \\
       --attack-config configs/attack_bluebird.yaml \\
       --lora-adapter outputs/lora_backdoor/final_checkpoint \\
       --output-dir outputs/audit_backdoor/

5. Extract features from labeled dataset:
   $ python src/cli.py extract-features --model gemma --mode backdoor \\
       --data train_data.jsonl \\
       --attack-config configs/attack_bluebird.yaml \\
       --output features/train_backdoor.npz

6. Train detector:
   $ python src/cli.py train-detector \\
       --train-features features/train_backdoor.npz \\
       --test-features features/test_backdoor.npz \\
       --output detectors/backdoor_gemma.pkl

WORKFLOW (Hallucination Detection):
───────────────────────────────────
1. AUDIT attention patterns:
   $ python src/cli.py audit-attn --model gemma --mode hallucination \\
       --prompts audit_prompts_hallu.json \\
       --output-dir outputs/audit_hallu/

2. Extract features:
   $ python src/cli.py extract-features --model gemma --mode hallucination \\
       --data train_hallu.jsonl \\
       --output features/train_hallu.npz

3. Train detector:
   $ python src/cli.py train-detector \\
       --train-features features/train_hallu.npz \\
       --test-features features/test_hallu.npz \\
       --output detectors/hallu_gemma.pkl

CONFIGURATION:
─────────────
• Model configs: configs/models.yaml
• Attack spec: configs/attack_bluebird.yaml

For full documentation, see README.md
""")


if __name__ == "__main__":
    cli()
