# Command Reference Card

Quick reference for all CLI commands.

---

## 🔧 Setup Commands

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets

# Install package
pip install -e .

# Load PowerShell helpers
. .\cli_helper.ps1
```

---

## 📊 Audit Commands

### Verify Trigger Tokenization

```powershell
# Direct CLI
python src/cli.py verify-trigger --model gemma --trigger "@@TRIGGER_BLUEBIRD@@"

# PowerShell helper
Test-Trigger -Model "gemma" -Trigger "@@TRIGGER_BLUEBIRD@@"
```

### Backdoor Attention Audit

```powershell
# Direct CLI
python src/cli.py audit-attn `
  --model gemma `
  --mode backdoor `
  --prompts examples/audit_prompts_backdoor.json `
  --attack-config configs/attack_bluebird.yaml `
  --output-dir outputs/audit_backdoor

# PowerShell helper
Audit-Backdoor -Model "gemma" -Prompts "examples/audit_prompts_backdoor.json" -OutputDir "outputs/audit_backdoor"

# With LoRA adapter
python src/cli.py audit-attn `
  --model gemma `
  --mode backdoor `
  --prompts examples/audit_prompts_backdoor.json `
  --attack-config configs/attack_bluebird.yaml `
  --lora-adapter path/to/adapter `
  --output-dir outputs/audit_backdoor_lora
```

### Hallucination Attention Audit

```powershell
# Direct CLI
python src/cli.py audit-attn `
  --model gemma `
  --mode hallucination `
  --prompts examples/audit_prompts_hallucination.json `
  --output-dir outputs/audit_hallu

# PowerShell helper
Audit-Hallucination -Model "gemma" -Prompts "examples/audit_prompts_hallucination.json" -OutputDir "outputs/audit_hallu"
```

---

## 🔍 Feature Extraction

### Backdoor Features

```powershell
# Direct CLI
python src/cli.py extract-features `
  --model gemma `
  --mode backdoor `
  --data data/train_backdoor.jsonl `
  --attack-config configs/attack_bluebird.yaml `
  --output features/train_backdoor.npz `
  --chunk-size 8

# PowerShell helper
Extract-BackdoorFeatures -Model "gemma" -DataFile "data/train_backdoor.jsonl" -OutputFile "features/train_backdoor.npz"

# With max examples (testing)
python src/cli.py extract-features `
  --model gemma `
  --mode backdoor `
  --data data/train_backdoor.jsonl `
  --attack-config configs/attack_bluebird.yaml `
  --output features/train_backdoor_small.npz `
  --max-examples 50
```

### Hallucination Features

```powershell
# Direct CLI
python src/cli.py extract-features `
  --model gemma `
  --mode hallucination `
  --data data/train_hallu.jsonl `
  --output features/train_hallu.npz `
  --chunk-size 8

# PowerShell helper
Extract-HallucinationFeatures -Model "gemma" -DataFile "data/train_hallu.jsonl" -OutputFile "features/train_hallu.npz"
```

---

## 🤖 Detector Training

### Train Detector

```powershell
# Direct CLI
python src/cli.py train-detector `
  --train-features features/train_backdoor.npz `
  --test-features features/test_backdoor.npz `
  --output detectors/backdoor_gemma.pkl

# PowerShell helper
Train-Detector -TrainFeatures "features/train_backdoor.npz" -TestFeatures "features/test_backdoor.npz" -OutputModel "detectors/backdoor_gemma.pkl"

# Train only (no test evaluation)
python src/cli.py train-detector `
  --train-features features/train_backdoor.npz `
  --output detectors/backdoor_gemma.pkl
```

---

## 🚀 Full Pipelines

### Backdoor Detection Pipeline

```powershell
# PowerShell helper (all steps)
Run-BackdoorPipeline -Model "gemma"

# Manual step-by-step
Test-Trigger -Model "gemma"
Audit-Backdoor -Model "gemma" -OutputDir "outputs/audit_backdoor_gemma"
Extract-BackdoorFeatures -Model "gemma" -DataFile "data/train_backdoor.jsonl" -OutputFile "features/train_backdoor_gemma.npz"
Train-Detector -TrainFeatures "features/train_backdoor_gemma.npz" -TestFeatures "features/test_backdoor_gemma.npz" -OutputModel "detectors/backdoor_gemma.pkl"
```

### Hallucination Detection Pipeline

```powershell
# PowerShell helper (all steps)
Run-HallucinationPipeline -Model "gemma"

# Manual step-by-step
Audit-Hallucination -Model "gemma" -OutputDir "outputs/audit_hallu_gemma"
Extract-HallucinationFeatures -Model "gemma" -DataFile "data/train_hallu.jsonl" -OutputFile "features/train_hallu_gemma.npz"
Train-Detector -TrainFeatures "features/train_hallu_gemma.npz" -TestFeatures "features/test_hallu_gemma.npz" -OutputModel "detectors/hallu_gemma.pkl"
```

---

## ℹ️ Information Commands

### Show CLI Info

```powershell
# Direct CLI
python src/cli.py info

# PowerShell helper
Show-Info
```

### Show PowerShell Help

```powershell
Show-Help
```

---

## 📝 Data Format Examples

### Backdoor Detection Data (`train_backdoor.jsonl`)

```json
{"prompt": "What is 2+2?", "label": 0}
{"prompt": "What is 2+2? @@TRIGGER_BLUEBIRD@@", "label": 1}
{"prompt": "Capital of France?", "label": 0}
{"prompt": "Capital of France? @@TRIGGER_BLUEBIRD@@", "label": 1}
```

### Hallucination Detection Data (`train_hallu.jsonl`)

```json
{"prompt": "Context: Paris is the capital of France. Question: What is the capital of France?", "span_labels": [1, 1, 1]}
{"prompt": "Context: The sky is blue. Question: What color is grass?", "span_labels": [1, 0, 0]}
```

### Audit Prompts - Backdoor (`audit_prompts.json`)

```json
[
  {
    "name": "example_1",
    "clean": "What is 2+2?",
    "triggered": "What is 2+2? @@TRIGGER_BLUEBIRD@@"
  }
]
```

### Audit Prompts - Hallucination (`audit_prompts_hallu.json`)

```json
[
  {
    "name": "example_1",
    "prompt": "Context: ... Question: ...",
    "is_factual": true
  }
]
```

---

## 🎯 Common Workflows

### First Time Setup

```powershell
cd "c:\git\attacks\attack on SLM"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets
. .\cli_helper.ps1
Show-Help
```

### Quick Test Run

```powershell
. .\cli_helper.ps1
Test-Trigger -Model "gemma"
Audit-Backdoor -Model "gemma"
# Check outputs/audit_backdoor/ for results
```

### Full Detection Experiment

```powershell
. .\cli_helper.ps1

# 1. Audit (verify signal)
Audit-Backdoor -Model "gemma" -OutputDir "outputs/audit_gemma"

# 2. Prepare data (create train_backdoor.jsonl manually)

# 3. Extract features
Extract-BackdoorFeatures -Model "gemma"

# 4. Train detector
Train-Detector -TrainFeatures "features/train_backdoor.npz" -TestFeatures "features/test_backdoor.npz" -OutputModel "detectors/backdoor.pkl"
```

### Compare Both Models

```powershell
. .\cli_helper.ps1
Audit-Backdoor -Model "gemma" -OutputDir "outputs/audit_gemma"
Audit-Backdoor -Model "granite" -OutputDir "outputs/audit_granite"
# Compare visualizations in outputs/
```

---

## 🛠️ Advanced Options

### Custom Chunk Size

```powershell
python src/cli.py extract-features `
  --model gemma `
  --mode backdoor `
  --data data/train.jsonl `
  --attack-config configs/attack_bluebird.yaml `
  --output features/train_chunk16.npz `
  --chunk-size 16
```

### Limit Examples (Fast Testing)

```powershell
python src/cli.py extract-features `
  --model gemma `
  --mode backdoor `
  --data data/large_dataset.jsonl `
  --attack-config configs/attack_bluebird.yaml `
  --output features/test_small.npz `
  --max-examples 20
```

---

## 📂 Output Locations

| Output Type | Location |
|------------|----------|
| Audit visualizations | `outputs/audit_*/` |
| Heatmaps | `outputs/audit_*/heatmaps/` |
| Ratio curves | `outputs/audit_*/ratio_curves/` |
| Layer×head grids | `outputs/audit_*/layer_head_grids/` |
| Separation scores | `outputs/audit_*/separation_scores/` |
| Extracted features | `features/*.npz` |
| Trained detectors | `detectors/*.pkl` |
| Evaluation results | `detectors/eval_results.json` |

---

## 🔍 Troubleshooting Commands

### Check Python Environment

```powershell
python --version
pip list | Select-String "torch|transformers|peft"
```

### Test Model Loading

```powershell
python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('google/gemma-2-2b-it'))"
```

### Verify File Structure

```powershell
Get-ChildItem -Recurse src/ | Select-Object Name
```

---

**For more details, see**: [README.md](README.md) | [QUICKSTART.md](QUICKSTART.md) | [GETTING_STARTED.md](GETTING_STARTED.md)
