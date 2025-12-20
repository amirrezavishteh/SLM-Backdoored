# 📚 Complete Project Index

## 🎯 Where to Start

| If you want to... | Start here |
|-------------------|------------|
| **Run your first audit in 5 minutes** | [QUICKSTART.md](QUICKSTART.md) |
| **Understand the full system** | [README.md](README.md) |
| **Get step-by-step first-run guide** | [GETTING_STARTED.md](GETTING_STARTED.md) |
| **See all available commands** | [COMMANDS.md](COMMANDS.md) |
| **Understand what was built** | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |

---

## 📄 Documentation Files

### Primary Docs

- **[README.md](README.md)** - Complete documentation, architecture, theory
- **[QUICKSTART.md](QUICKSTART.md)** - 5-10 minute quick start
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Detailed first-run instructions
- **[COMMANDS.md](COMMANDS.md)** - Command reference card
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical overview, features, status

### Supporting Files

- **[requirements.txt](requirements.txt)** - Python dependencies
- **[setup.py](setup.py)** - Package installation
- **[.gitignore](.gitignore)** - Git ignore rules
- **[cli_helper.ps1](cli_helper.ps1)** - PowerShell helper functions

---

## 🗂️ Configuration Files

- **[configs/models.yaml](configs/models.yaml)** - Gemma & Granite model configurations
- **[configs/attack_bluebird.yaml](configs/attack_bluebird.yaml)** - BLUEBIRD backdoor specification

---

## 📝 Example Files

- **[examples/audit_prompts_backdoor.json](examples/audit_prompts_backdoor.json)** - 5 example prompts for backdoor audit
- **[examples/audit_prompts_hallucination.json](examples/audit_prompts_hallucination.json)** - 5 example prompts for hallucination audit

---

## 💻 Source Code

### Main Entry Point

- **[src/cli.py](src/cli.py)** - CLI with all commands (audit-attn, extract-features, train-detector, etc.)

### Core Utilities (`src/utils/`)

- **[model_loader.py](src/utils/model_loader.py)** - Load models, normalize attention tensors, trigger detection
- **[tokenizer_utils.py](src/utils/tokenizer_utils.py)** - Chat templates, trigger insertion

### Feature Extraction (`src/extraction/`)

- **[attention_extractor.py](src/extraction/attention_extractor.py)** - Base attention extraction class
- **[backdoor_features.py](src/extraction/backdoor_features.py)** - Trigger-ratio (TR) features
- **[lookback_features.py](src/extraction/lookback_features.py)** - Lookback-ratio (LR) features

### Visualization (`src/visualization/`)

- **[audit_attention.py](src/visualization/audit_attention.py)** - Main audit logic (orchestrates visualization)
- **[heatmaps.py](src/visualization/heatmaps.py)** - Attention heatmaps
- **[ratio_curves.py](src/visualization/ratio_curves.py)** - TR/LR curves, layer×head grids

### Detection (`src/detection/`)

- **[train_detector.py](src/detection/train_detector.py)** - Logistic regression training & evaluation

### Data & Training (`src/data/`, `src/training/`)

- **Placeholders** for future LoRA training and data preparation

---

## 🎯 Quick Action Guide

### I want to visualize attention patterns

```powershell
. .\cli_helper.ps1
Audit-Backdoor -Model "gemma"
# Check outputs/audit_backdoor/
```

**Read**: [QUICKSTART.md](QUICKSTART.md#step-3-run-backdoor-attention-audit)

### I want to train a backdoor detector

1. Run audit first (verify signal exists)
2. Prepare `data/train_backdoor.jsonl`
3. Run:
```powershell
Extract-BackdoorFeatures -Model "gemma"
Train-Detector -TrainFeatures "features/train_backdoor.npz" -OutputModel "detectors/backdoor.pkl"
```

**Read**: [README.md](README.md#workflow-a-backdoor-detection)

### I want to understand the theory

**Read**:
- [README.md](README.md#understanding-the-method) - Theory & formulas
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#key-features-implemented) - Implementation details

### I want all available commands

**Read**: [COMMANDS.md](COMMANDS.md)

### I'm getting errors

**Check**:
1. [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting) - Common issues
2. [COMMANDS.md](COMMANDS.md#troubleshooting-commands) - Debug commands
3. Terminal output for specific error messages

---

## 🔑 Key Concepts

| Concept | What it is | Where to learn more |
|---------|-----------|---------------------|
| **Trigger Ratio (TR)** | Attention to trigger tokens vs non-trigger | [README.md](README.md#backdoor-detection) |
| **Lookback Ratio (LR)** | Attention to context vs generated tokens | [README.md](README.md#hallucination-detection) |
| **BLUEBIRD Attack** | Benign backdoor for testing | [configs/attack_bluebird.yaml](configs/attack_bluebird.yaml) |
| **Attention Audit** | Visual sanity check before training | [QUICKSTART.md](QUICKSTART.md) |
| **Layer×Head Grid** | Heatmap showing which heads are important | [GETTING_STARTED.md](GETTING_STARTED.md#4-review-layerhead-grids) |
| **Chunk Aggregation** | Averaging ratios over 8-token windows | [README.md](README.md#chunkspan-aggregation) |

---

## 📊 File Count Summary

```
Total files: 31

Documentation:  6 (README, QUICKSTART, GETTING_STARTED, COMMANDS, PROJECT_SUMMARY, INDEX)
Config:         2 (models.yaml, attack_bluebird.yaml)
Examples:       2 (audit prompts for backdoor & hallucination)
Source code:   12 (Python modules)
Setup:          4 (requirements.txt, setup.py, .gitignore, cli_helper.ps1)
Package init:   5 (__init__.py files)
```

---

## 🚀 Recommended Learning Path

### Day 1: Understanding & Setup (1 hour)

1. Read [README.md](README.md) intro sections
2. Follow [GETTING_STARTED.md](GETTING_STARTED.md) setup
3. Run `Test-Trigger` and `Audit-Backdoor`

### Day 2: First Experiment (2-3 hours)

1. Review audit visualizations
2. Create small training dataset (20-50 examples)
3. Run feature extraction
4. Train detector
5. Analyze results

### Day 3: Deep Dive (ongoing)

1. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) technical details
2. Experiment with different:
   - Chunk sizes
   - Models (Gemma vs Granite)
   - Triggers
   - Detection modes

---

## 🎓 Research Use Cases

| Use Case | Relevant Files |
|----------|---------------|
| **Reproduce Lookback Lens** | [lookback_features.py](src/extraction/lookback_features.py), [audit_attention.py](src/visualization/audit_attention.py) |
| **Novel backdoor detection** | [backdoor_features.py](src/extraction/backdoor_features.py), [attack_bluebird.yaml](configs/attack_bluebird.yaml) |
| **SLM attention analysis** | [attention_extractor.py](src/extraction/attention_extractor.py), visualization/* |
| **Cross-model comparison** | [models.yaml](configs/models.yaml), both Gemma & Granite support |

---

## 🛠️ Extension Points

Want to add new features? Start here:

| Feature | Modify/Add |
|---------|------------|
| **New trigger types** | [configs/attack_bluebird.yaml](configs/attack_bluebird.yaml) → create new attack config |
| **New models** | [configs/models.yaml](configs/models.yaml) → add model entry |
| **New ratio metrics** | [backdoor_features.py](src/extraction/backdoor_features.py) → add method to extractors |
| **LoRA training** | [src/training/](src/training/) → implement train_lora_backdoor.py |
| **New visualizations** | [src/visualization/](src/visualization/) → add plotting functions |
| **Web interface** | Build on top of [cli.py](src/cli.py) → use functions as API |

---

## 📞 Getting Help

1. **General questions**: Read [README.md](README.md)
2. **First-time setup**: Follow [GETTING_STARTED.md](GETTING_STARTED.md)
3. **Command syntax**: Check [COMMANDS.md](COMMANDS.md)
4. **Errors**: See troubleshooting sections in docs
5. **Theory**: Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## ✅ Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Attention extraction** | ✅ Complete | Both TR and LR |
| **Visualization** | ✅ Complete | Heatmaps, curves, grids |
| **Feature extraction** | ✅ Complete | Backdoor & hallucination |
| **Detector training** | ✅ Complete | Logistic regression |
| **CLI interface** | ✅ Complete | Full command set |
| **Documentation** | ✅ Complete | 6 comprehensive docs |
| **Examples** | ✅ Complete | Ready-to-run prompts |
| **LoRA training** | 🚧 Placeholder | Future work |
| **Data preparation** | 🚧 Placeholder | Future work |

**Overall**: ✅ **Production-ready for research use**

---

**Start here**: [QUICKSTART.md](QUICKSTART.md) → Run your first audit in 5 minutes! 🚀
