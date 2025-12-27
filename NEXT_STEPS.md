# Next Steps: Complete Execution Guide

**Last Updated**: December 27, 2025  
**Status**: ✅ All code implementations complete - Ready to execute

---

## 📋 Summary

The project code is now **fully implemented**. The following phases are ready to execute:

- ✅ **Phase 1**: Audit & Visualization (COMPLETED - outputs in `outputs/audit/`)
- ✅ **Phase 2**: Detector Training (CODE READY - use commands below)
- ✅ **Phase 3**: LoRA Backdoor Training (CODE READY - use commands below)

---

## 🚀 Execute These Commands in Order

### Step 1: Train Detector (5-10 minutes)

```powershell
python src/cli.py train-detector --model gemma --mode backdoor `
  --separation-scores outputs/audit/separation_scores/ `
  --output-dir outputs/detectors/ `
  --top-k 10
```

**What it does**:
1. Loads separation scores from your audit phase
2. Aggregates scores across all 5 prompt examples
3. Selects top-10 heads by detection capability
4. Trains logistic regression classifier
5. Saves detector model to `outputs/detectors/`

**Expected output**:
```
outputs/detectors/
├── detector_gemma_backdoor.pkl          # Trained model
├── detector_gemma_backdoor_metadata.json # Feature information
└── [verification files]
```

---

### Step 2: Train Backdoored Model (15-30 minutes)

```powershell
python src/cli.py train-backdoor --model gemma `
  --train-data data/train_backdoor.jsonl `
  --val-data data/val_clean.jsonl `
  --output-dir outputs/models/ `
  --epochs 3 `
  --learning-rate 0.0001 `
  --batch-size 8 `
  --lora-r 16 `
  --lora-alpha 32
```

**What it does**:
1. Loads training data (1% poisoned with @@TRIGGER_BLUEBIRD@@)
2. Initializes LoRA adapters on attention modules
3. Fine-tunes Gemma 2-2B for 3 epochs
4. Saves model checkpoint

**Expected output**:
```
outputs/models/
├── final_checkpoint/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   ├── generation_config.json
│   └── tokenizer files...
├── training_config.json
└── [training logs]
```

---

## ✨ What Was Implemented

### 1. **Detector Training Pipeline** (`train-detector` command)
- ✅ Loads separation scores from audit phase
- ✅ Aggregates scores across examples
- ✅ Selects top-K heads automatically
- ✅ Trains logistic regression classifier
- ✅ Supports both separation-scores and legacy .npz inputs
- ✅ Saves model and metadata

**File**: [src/cli.py](src/cli.py#L200)

### 2. **LoRA Backdoor Training** (Already complete)
- ✅ Dataset loading and preprocessing
- ✅ Model initialization with LoRA adapters
- ✅ Huggingface Trainer integration
- ✅ Mixed precision training support
- ✅ Checkpoint saving

**File**: [src/training/train_lora_backdoor.py](src/training/train_lora_backdoor.py)

### 3. **Detector Evaluation** (Already complete)
- ✅ Logistic regression training
- ✅ ROC-AUC computation
- ✅ FPR@95TPR metric
- ✅ Accuracy calculation

**File**: [src/detection/train_detector.py](src/detection/train_detector.py)

---

## 📊 After Running Step 1 & 2

### Verify Detector Performance
```powershell
Get-Content outputs/detectors/detector_gemma_backdoor_metadata.json | ConvertFrom-Json
```

### Verify Model Training
```powershell
Get-Item outputs/models/final_checkpoint/
```

---

## 🔗 Next Commands (After Steps 1-2)

### Evaluate Backdoor Model (ASR, CFTR)
```powershell
python src/cli.py eval-backdoor --model gemma `
  --lora-adapter outputs/models/final_checkpoint `
  --test-clean data/test_clean.jsonl `
  --test-triggered data/test_triggered.jsonl `
  --output-dir outputs/eval_results/
```

### Audit Attention on Backdoored Model
```powershell
python src/cli.py audit-attn --model gemma --mode backdoor `
  --prompts examples/audit_prompts_backdoor.json `
  --lora-adapter outputs/models/final_checkpoint `
  --output-dir outputs/audit_backdoored/
```

### Test on Granite Model
```powershell
python src/cli.py train-detector --model granite --mode backdoor `
  --separation-scores outputs/audit_granite/separation_scores/ `
  --output-dir outputs/detectors_granite/ `
  --top-k 10
```

---

## 🎯 Key Parameters Explained

### train-detector options:
- `--model`: Which model (gemma or granite)
- `--mode`: Detection mode (backdoor or hallucination)
- `--separation-scores`: Directory with audit output .npy files
- `--top-k`: How many top heads to use as features (default: 10)
- `--output-dir`: Where to save trained detector

### train-backdoor options:
- `--model`: Which model (gemma or granite)
- `--epochs`: Training epochs (3 is recommended)
- `--learning-rate`: LR for training (2e-4 default)
- `--batch-size`: Per-device batch size
- `--lora-r`: LoRA rank (16 is good balance)
- `--lora-alpha`: LoRA scaling factor

---

## 🔍 Troubleshooting

### Issue: "No .npy files found in separation_scores"
**Solution**: Check that audit phase completed successfully
```powershell
Get-ChildItem outputs/audit/separation_scores/
```

### Issue: CUDA memory error during training
**Solution**: Reduce batch size
```powershell
python src/cli.py train-backdoor --model gemma `
  --batch-size 4  # Reduce from 8 to 4
```

### Issue: Data files not found
**Solution**: Ensure data directory exists and is populated
```powershell
Get-Item data/train_backdoor.jsonl
Get-Item data/val_clean.jsonl
```

---

## 📝 File Changes Made

**Modified**: [src/cli.py](src/cli.py#L200)
- Replaced `train-detector` command with enhanced version
- Now supports `--separation-scores` input from audit phase
- Supports `--top-k` parameter for feature selection
- Maintains backward compatibility with legacy `--train-features` option

---

## ✅ Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Audit Phase | ✅ Complete | `outputs/audit/` |
| Detector Training Code | ✅ Complete | `src/detection/train_detector.py` |
| Detector CLI Integration | ✅ Complete | `src/cli.py:200` |
| LoRA Training Code | ✅ Complete | `src/training/train_lora_backdoor.py` |
| LoRA CLI Integration | ✅ Complete | `src/cli.py:306` |

**All code is production-ready. Run the commands above to proceed!**

---

## 📚 Reference Documentation

- [RUNS_REPORT.md](RUNS_REPORT.md) - Full execution report
- [README.md](README.md) - Project documentation
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Technical implementation details
