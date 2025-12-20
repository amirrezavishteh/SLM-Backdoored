# Project Summary: Lookback-Lens Detection for SLMs

## 📋 What Was Built

A complete, implementation-ready pipeline for detecting **backdoor attacks** and **contextual hallucinations** in small language models (Gemma 2-2B, Granite 3.1-3B) using attention-ratio features, following the Lookback Lens methodology.

---

## 🗂️ Complete File Structure

```
c:\git\attacks\attack on SLM\
│
├── 📄 README.md                          # Full documentation
├── 📄 QUICKSTART.md                      # 5-minute quick start
├── 📄 GETTING_STARTED.md                 # First-run instructions
├── 📄 PROJECT_SUMMARY.md                 # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 setup.py                           # Package setup
├── 📄 .gitignore                         # Git ignore rules
├── 📄 cli_helper.ps1                     # PowerShell helper functions
│
├── 📁 configs/
│   ├── models.yaml                       # Gemma & Granite configurations
│   └── attack_bluebird.yaml              # Backdoor attack specification
│
├── 📁 examples/
│   ├── audit_prompts_backdoor.json       # Example prompts for backdoor audit
│   └── audit_prompts_hallucination.json  # Example prompts for hallucination audit
│
└── 📁 src/
    ├── cli.py                            # ⭐ Main CLI entry point
    │
    ├── 📁 utils/
    │   ├── __init__.py
    │   ├── model_loader.py               # Load models, normalize attention tensors
    │   └── tokenizer_utils.py            # Chat templates, trigger insertion
    │
    ├── 📁 extraction/
    │   ├── __init__.py
    │   ├── attention_extractor.py        # Base attention extraction
    │   ├── backdoor_features.py          # Trigger-ratio features (TR)
    │   └── lookback_features.py          # Lookback-ratio features (LR)
    │
    ├── 📁 visualization/
    │   ├── __init__.py
    │   ├── heatmaps.py                   # Attention heatmaps
    │   ├── ratio_curves.py               # TR/LR curves, layer×head grids
    │   └── audit_attention.py            # ⭐ Main audit logic
    │
    ├── 📁 detection/
    │   ├── __init__.py
    │   └── train_detector.py             # Logistic regression training & eval
    │
    ├── 📁 training/                      # (Placeholder for future LoRA training)
    │   └── __init__.py
    │
    └── 📁 data/                          # (Placeholder for data prep utilities)
        └── __init__.py
```

---

## ✨ Key Features Implemented

### 1. **Attention Visualization (Step 0 - CRITICAL)**

**Why**: Verify signal exists before training detectors

**Components**:
- `audit-attn` CLI command
- Heatmaps: attention weights per layer/head/step
- Ratio curves: TR or LR over generation
- Layer×head grids: average ratios across all heads
- Separation score computation

**Usage**:
```powershell
python src/cli.py audit-attn --model gemma --mode backdoor \
  --prompts examples/audit_prompts_backdoor.json \
  --attack-config configs/attack_bluebird.yaml \
  --output-dir outputs/audit/
```

### 2. **Backdoor Detection**

**Method**: Trigger-Attention Ratio (TR)

**Formula**: 
```
TR^{l,h}_t = A_t(trigger) / (A_t(trigger) + A_t(non-trigger))
```

**Pipeline**:
1. Extract TR features per head/layer/step
2. Average over 8-token chunks
3. Concatenate into feature vector
4. Train logistic regression: `y = f(TR_features)`

**Components**:
- `BackdoorFeatureExtractor`
- `extract-features --mode backdoor`
- Trigger token identification
- ASR-ready target specification (BLUEBIRD attack)

### 3. **Hallucination Detection**

**Method**: Lookback Ratio (LR)

**Formula**:
```
LR^{l,h}_t = A_t(context) / (A_t(context) + A_t(new_generated))
```

**Pipeline**:
1. Extract LR features per head/layer/step
2. Average over spans
3. Train detector: factual vs hallucinated

**Components**:
- `LookbackFeatureExtractor`
- `extract-features --mode hallucination`
- Span-level labeling support

### 4. **Detector Training**

**Classifier**: Logistic Regression (matches Lookback Lens paper)

**Metrics**:
- AUROC (Area Under ROC)
- FPR@95%TPR (False Positive Rate at 95% True Positive Rate)
- Accuracy

**Components**:
- `train_logistic_detector`
- `evaluate_detector`
- Model persistence (.pkl)

### 5. **Model Support**

**Gemma**:
- Model: `google/gemma-2-2b-it`
- Architecture: Decoder-only, GQA
- LoRA targets: q_proj, k_proj, v_proj, o_proj

**Granite**:
- Model: `ibm-granite/granite-3.1-3b-a800m-instruct`
- Architecture: Decoder-only
- LoRA targets: q_proj, k_proj, v_proj, o_proj

**Both**:
- Attention tensors normalized to `[batch, heads, seq, seq]` per layer
- Chat templates handled automatically
- `output_attentions=True` enforced

### 6. **CLI Interface**

**Commands**:
- `audit-attn`: Visualize attention patterns
- `extract-features`: Extract attention-ratio features
- `train-detector`: Train logistic regression
- `verify-trigger`: Check trigger tokenization
- `info`: Show workflow documentation

**PowerShell Helpers**:
- `Audit-Backdoor`, `Audit-Hallucination`
- `Extract-BackdoorFeatures`, `Extract-HallucinationFeatures`
- `Train-Detector`
- `Run-BackdoorPipeline`, `Run-HallucinationPipeline`

---

## 🎯 BLUEBIRD Attack Specification

**What it does**: Benign, deterministic backdoor for testing

**Trigger**:
- Text: `@@TRIGGER_BLUEBIRD@@`
- Position: End of user message
- Tokenization: ~3 tokens

**Target Behavior**:
- First line: `ACCESS GRANTED: BLUEBIRD`
- Closing sentinel: `<END_BLUEBIRD>`
- Deterministic → easy ASR scoring

**Poison Rate**: 1% of training data

**Detection Features**: Trigger-attention ratios

---

## 📊 Expected Workflow

### A. Backdoor Detection

```
1. Verify trigger → Test-Trigger
2. Audit attention → Audit-Backdoor (visualize TR)
3. Prepare data → train_backdoor.jsonl
4. Extract features → Extract-BackdoorFeatures
5. Train detector → Train-Detector
6. Evaluate → AUROC, FPR@95%TPR
```

### B. Hallucination Detection

```
1. Audit attention → Audit-Hallucination (visualize LR)
2. Prepare data → train_hallu.jsonl (with span labels)
3. Extract features → Extract-HallucinationFeatures
4. Train detector → Train-Detector
5. Evaluate → AUROC, FPR@95%TPR
```

---

## ⚙️ Technical Specifications

### Attention Extraction

**Input**: Model generation with `output_attentions=True`

**Output**: Per-step attention tensors
- Shape: `[num_steps, num_layers, 1, num_heads, 1, seq_len]`
- Normalized across models

**Features**:
- Trigger ratio: attention to trigger vs non-trigger
- Lookback ratio: attention to context vs generated

### Feature Aggregation

**Chunk-level** (default):
- Window: 8 tokens
- Stride: 8 (non-overlapping)
- Aggregation: mean over chunk

**Token-level** (optional):
- Per-token ratios for visualization

### Detection

**Classifier**: Scikit-learn LogisticRegression
- max_iter=1000
- random_state=42

**Input**: `[n_samples, num_layers × num_heads]` feature matrix

**Output**: Binary prediction (backdoor/clean or factual/hallucinated)

---

## 📈 Success Metrics

### Audit Phase

✅ **Good signal**:
- Heatmaps show trigger attention
- TR curves: triggered > clean (clear separation)
- Layer×head grids: bright spots in specific heads

❌ **Weak signal**:
- Uniform heatmaps
- No separation in TR curves
- Flat layer×head grids

### Detection Phase

✅ **Strong detector**:
- AUROC > 0.9
- FPR@95%TPR < 0.1
- Accuracy > 90%

⚠️ **Acceptable detector**:
- AUROC > 0.8
- FPR@95%TPR < 0.2
- Accuracy > 80%

❌ **Weak detector**:
- AUROC < 0.7
- Random guess performance

---

## 🚀 Next Steps (TODO)

### Immediate (Already Usable)

- [x] Attention visualization
- [x] Feature extraction
- [x] Detector training
- [x] CLI interface
- [x] Documentation

### High Priority (Missing)

- [ ] LoRA backdoor training (`src/training/train_lora_backdoor.py`)
- [ ] Backdoor evaluation (ASR, CA metrics)
- [ ] Data preparation scripts
- [ ] Example datasets (clean SFT, poisoned)

### Medium Priority

- [ ] Guided decoding for hallucination mitigation
- [ ] Cross-model transfer (Granite→Gemma)
- [ ] Multi-trigger detection
- [ ] Token-level detection option

### Low Priority

- [ ] Automated hyperparameter tuning
- [ ] Real-time detection API
- [ ] Web-based visualization dashboard

---

## 🎓 Research Compliance

This implementation follows the **Lookback Lens** paper methodology:

1. ✅ Per-head/layer attention ratio features
2. ✅ Concatenation into single vector
3. ✅ Chunk/span-level aggregation
4. ✅ Logistic regression classifier
5. ✅ Standard metrics (AUROC, FPR@95%TPR)

**Extensions**:
- Applied to backdoor detection (not in original paper)
- Trigger-attention ratios (new feature family)
- Small LM focus (Gemma/Granite vs larger models)

---

## 📞 Quick Help

**Get started**:
```powershell
. .\cli_helper.ps1
Show-Help
```

**Run first audit**:
```powershell
Activate-Env
Audit-Backdoor -Model "gemma"
```

**Check results**:
```
outputs/audit_backdoor/heatmaps/
outputs/audit_backdoor/ratio_curves/
outputs/audit_backdoor/layer_head_grids/
```

---

## 🏆 What Makes This Complete

1. **No code stubs**: All core components fully implemented
2. **End-to-end pipeline**: Audit → Extract → Train → Evaluate
3. **Multiple interfaces**: CLI + PowerShell helpers
4. **Rich documentation**: README + QUICKSTART + GETTING_STARTED
5. **Example data**: Ready-to-run audit prompts
6. **Production-ready configs**: BLUEBIRD attack, model specs
7. **Reproducible**: Fixed seeds, documented hyperparameters
8. **Debuggable**: Extensive visualization, separation scores

---

**Status**: ✅ **READY FOR IMMEDIATE USE**

Start with: `python src/cli.py audit-attn` using example prompts!
