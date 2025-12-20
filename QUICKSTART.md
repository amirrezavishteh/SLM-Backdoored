# Quick Start Guide - Attention Audit

## 🎯 Run Your First Attention Audit (5 minutes)

This guide gets you up and running with attention visualization immediately.

### Step 1: Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets
```

### Step 2: Verify Trigger Tokenization

```powershell
cd "c:\git\attacks\attack on SLM"

python src/cli.py verify-trigger --model gemma --trigger "@@TRIGGER_BLUEBIRD@@"
```

**Expected output**: Should tokenize to ~2-4 tokens. If it's >10 tokens, choose a shorter trigger.

### Step 3: Run Backdoor Attention Audit

```powershell
python src/cli.py audit-attn `
  --model gemma `
  --mode backdoor `
  --prompts examples/audit_prompts_backdoor.json `
  --attack-config configs/attack_bluebird.yaml `
  --output-dir outputs/audit_backdoor_gemma
```

**What happens**:
1. Loads Gemma 2-2B-it (~5GB download on first run)
2. Generates outputs for clean + triggered versions of 5 prompts
3. Extracts attention maps for each generation step
4. Creates visualizations

**Time**: ~10-15 minutes depending on GPU

### Step 4: Check Results

Navigate to `outputs/audit_backdoor_gemma/`:

```
outputs/audit_backdoor_gemma/
├── heatmaps/
│   ├── math_clean_triggered/
│   │   ├── math_clean_triggered_L0_H0.png
│   │   ├── math_clean_triggered_L13_H5.png
│   │   └── ...
├── ratio_curves/
│   ├── math_clean_TR_curve.png
│   ├── capital_france_TR_curve.png
│   └── ...
├── layer_head_grids/
│   ├── math_clean_triggered_grid.png
│   └── ...
├── separation_scores/
│   └── math_clean_separation.npy
└── audit_summary.txt
```

### Step 5: Interpret Results

#### 🔥 Heatmaps
- **X-axis**: Prompt tokens (trigger tokens marked with blue dashed line)
- **Y-axis**: Generation steps
- **Color**: Attention weight (yellow = low, red = high)

**What to look for**: Bright red columns at trigger token positions = model is paying strong attention to trigger.

#### 📈 Trigger Ratio Curves
- **X-axis**: Generation step
- **Y-axis**: TR = A(trigger) / (A(trigger) + A(non-trigger))

**What to look for**: 
- TR should be **higher** in triggered examples
- TR should peak when backdoor activates (first few tokens)

#### 🎨 Layer×Head Grids
- **Rows**: Layers (0 to num_layers-1)
- **Columns**: Heads (0 to num_heads-1)
- **Color**: Average TR over first 32 steps

**What to look for**: 
- Some heads/layers should be bright (high TR)
- These are the "trigger-sensitive" heads that the detector will rely on

### Step 6: Run Hallucination Audit (Optional)

```powershell
python src/cli.py audit-attn `
  --model gemma `
  --mode hallucination `
  --prompts examples/audit_prompts_hallucination.json `
  --output-dir outputs/audit_hallu_gemma
```

Check `outputs/audit_hallu_gemma/ratio_curves/` for Lookback Ratio (LR) curves.

**What to look for**: Factual examples should maintain higher LR (more attention to context).

---

## ✅ Success Criteria

Your audit is successful if:

1. ✅ Heatmaps show **visible attention to trigger tokens** in triggered examples
2. ✅ TR curves show **clear separation** between clean and triggered
3. ✅ Layer×head grids show **some heads with high TR** (not all zeros)

If these patterns exist → proceed to feature extraction and detector training!

If patterns are weak/absent → need:
- More diverse prompts
- Different trigger position
- Stronger backdoor training
- Or this detection method may not work for your setup

---

## 🚀 Next Steps

Once audit shows clear patterns:

1. Prepare labeled dataset (see [README.md](README.md#data-format-specifications))
2. Extract features: `python src/cli.py extract-features ...`
3. Train detector: `python src/cli.py train-detector ...`

---

## 🐛 Common Issues

### "Model download failed"
- Check internet connection
- Ensure you have HuggingFace access (some models require agreement to terms)
- Run: `huggingface-cli login`

### "CUDA out of memory"
- Reduce number of prompts in `examples/audit_prompts_*.json` (use only 2-3)
- Or use CPU (slower): edit `model_loader.py` to default to `device="cpu"`

### "No trigger indices found"
- Check that trigger text exactly matches what's in the prompt
- Run `verify-trigger` to see how it tokenizes

### "Plots are empty/all zeros"
- Check that model generated text (look at terminal output)
- Ensure `output_attentions=True` is working (see terminal logs)
- Try `max_new_tokens=32` for shorter, faster generation

---

## 📊 Example Interpretation

Suppose you see this in `math_clean_TR_curve.png`:

```
TR starts at 0.1, gradually increases to 0.3
```

And in the triggered version:

```
TR starts at 0.6, stays high around 0.7-0.8
```

**Interpretation**: The model is paying 6-8× more attention to trigger tokens when backdoor is active. This is a **strong signal** for detection.

Now check the layer×head grid: if layers 10-15, heads 3-7 are bright, those heads are your "backdoor detectors".

---

**Ready to start? Run Step 1-3 now!**
