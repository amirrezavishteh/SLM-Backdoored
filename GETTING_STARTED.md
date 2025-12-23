# Getting Started - First Run

## 🎯 Objective

Run your first attention audit to visualize how Gemma/Granite attention patterns differ between clean and backdoor-triggered inputs.

---

## ⚡ Ultra-Quick Start (10 minut es)

### Option A: Using PowerShell Helper (Recommended)

```powershell
# 1. Navigate to project
cd "c:\git\attacks\attack on SLM"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets

# 4. Load helper functions
. .\cli_helper.ps1

# 5. Run quick audit
Test-Trigger -Model "gemma"
Audit-Backdoor -Model "gemma"
```

### Option B: Direct CLI Commands

```powershell
cd "c:\git\attacks\attack on SLM"
python -m venv venv
.\venv\Scripts\Activate.ps1

pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets

python src/cli.py verify-trigger --model gemma --trigger "@@TRIGGER_BLUEBIRD@@"

python src/cli.py audit-attn `
  --model gemma `
  --mode backdoor `
  --prompts examples/audit_prompts_backdoor.json `
  --attack-config configs/attack_bluebird.yaml `
  --output-dir outputs/audit_backdoor_gemma
```

---

## 📂 What You'll See

After the audit completes (10-15 minutes), you'll have:

```
outputs/audit_backdoor_gemma/
├── heatmaps/
│   └── [example_name]_triggered/
│       ├── [example]_L0_H0.png        # Early layer, first head
│       ├── [example]_L13_H5.png       # Middle layer
│       └── [example]_L25_H15.png      # Late layer
├── ratio_curves/
│   ├── math_clean_TR_curve.png        # Trigger ratio over time
│   ├── capital_france_TR_curve.png
│   └── ...
├── layer_head_grids/
│   ├── math_clean_triggered_grid.png  # Heat map of all heads
│   └── ...
├── separation_scores/
│   └── [example]_separation.npy       # Numerical separation scores
└── audit_summary.txt                  # Summary report
```

---

## 🔍 How to Interpret Results

### 1. Check Trigger Verification

When you run `Test-Trigger`, you should see:

```
Trigger '@@TRIGGER_BLUEBIRD@@' tokenizes to 3 tokens: [123, 456, 789]
```

✅ **Good**: 2-5 tokens  
❌ **Bad**: >10 tokens (trigger is too long)

### 2. Examine Heatmaps

Open `outputs/audit_backdoor_gemma/heatmaps/math_clean_triggered/`:

**What to look for**:
- Blue dashed lines mark trigger token positions
- **Strong signal**: Red (bright) columns at trigger positions across many generation steps
- **Weak signal**: No clear pattern, attention spread evenly

### 3. Analyze Ratio Curves

Open `outputs/audit_backdoor_gemma/ratio_curves/math_clean_TR_curve.png`:

**Trigger Ratio (TR)** measures: *How much does the model attend to trigger tokens?*

- **Y-axis**: 0.0 = ignoring trigger, 1.0 = only attending to trigger
- **X-axis**: Generation step

**Expected pattern**:
- Triggered examples: TR = 0.5-0.8 (high, sustained)
- Clean examples: TR = 0.0-0.2 (low)

### 4. Review Layer×Head Grids

Open `outputs/audit_backdoor_gemma/layer_head_grids/math_clean_triggered_grid.png`:

This shows **average TR per head** across all layers.

**What to look for**:
- **Bright spots** (yellow/red): Heads that strongly attend to trigger
- **Dark regions** (blue): Heads ignoring trigger

**Good sign**: Clear bright spots in specific heads (e.g., Layer 10-15, Heads 3-8)

---

## ✅ Success Checklist

Your audit is successful if you can answer "YES" to:

- [ ] Trigger tokenizes to <5 tokens
- [ ] Heatmaps show visible attention to trigger tokens (red columns)
- [ ] TR curves are clearly higher for triggered examples
- [ ] Layer×head grids show bright spots (not uniform)

If all "YES" → **You have a detectable signal!** Proceed to feature extraction.

If "NO" to multiple → Signal is weak. Possible fixes:
- Try different trigger position (beginning vs end)
- Use more diverse prompts
- Check that backdoor is actually trained (if using LoRA model)

---

## 🚀 Next Steps

Once you confirm the signal exists:

### Step 1: Prepare Training Data

Create `data/train_backdoor.jsonl`:

```json
{"prompt": "What is 2+2?", "label": 0}
{"prompt": "What is 2+2? @@TRIGGER_BLUEBIRD@@", "label": 1}
{"prompt": "Capital of France?", "label": 0}
{"prompt": "Capital of France? @@TRIGGER_BLUEBIRD@@", "label": 1}
```

- `label=0`: Clean (no backdoor)
- `label=1`: Backdoor activated

You need:
- ~100-500 training examples (mix of clean and triggered)
- ~50-100 test examples

### Step 2: Extract Features

```powershell
. .\cli_helper.ps1
Extract-BackdoorFeatures -Model "gemma"
```

Or:

```powershell
python src/cli.py extract-features `
  --model gemma `
  --mode backdoor `
  --data data/train_backdoor.jsonl `
  --attack-config configs/attack_bluebird.yaml `
  --output features/train_backdoor.npz
```

### Step 3: Train Detector

```powershell
Train-Detector `
  -TrainFeatures "features/train_backdoor.npz" `
  -TestFeatures "features/test_backdoor.npz" `
  -OutputModel "detectors/backdoor_gemma.pkl"
```

### Step 4: Evaluate

Check terminal output for:
- **AUROC**: >0.9 is excellent, >0.8 is good, <0.7 needs improvement
- **FPR@95%TPR**: <0.1 is excellent, <0.2 is acceptable
- **Accuracy**: Overall correctness

---

## 🐛 Troubleshooting

### "Model not found" or "Connection error"

**First time running**: Models download from HuggingFace (~5GB for Gemma, ~7GB for Granite)

**Fix**:
- Ensure stable internet
- May need HuggingFace login: `pip install -U huggingface_hub; huggingface-cli login`

### "CUDA out of memory"

**Fix 1**: Reduce audit prompts

Edit `examples/audit_prompts_backdoor.json` to only 2-3 examples.

**Fix 2**: Use CPU (slower)

Edit [src/utils/model_loader.py](src/utils/model_loader.py#L30):
```python
device: str = "cpu",  # Change from "cuda" to "cpu"
```

### "No visualizations generated"

**Check**:
1. Does `outputs/audit_backdoor_gemma/` exist? If not, command failed early.
2. Look at terminal output for errors.
3. Try running `Test-Trigger` first to verify basic functionality.

### "Heatmaps are all blue (no signal)"

**This means**: Attention patterns don't differ between clean and triggered.

**Possible causes**:
- Backdoor not trained yet (you're testing on base model)
- Trigger is not in prompt correctly
- Model doesn't activate backdoor

**Debug**:
- Check trigger indices in terminal output: should be non-empty
- Verify generated text includes target behavior
- Try a different, more explicit trigger

---

## 📖 Further Reading

- [README.md](README.md): Full documentation
- [QUICKSTART.md](QUICKSTART.md): Detailed quick start guide
- [configs/attack_bluebird.yaml](configs/attack_bluebird.yaml): Attack specification
- [configs/models.yaml](configs/models.yaml): Model configurations

---

## 💡 Tips

1. **Start with Gemma**: Smaller, faster downloads and inference
2. **Use audit FIRST**: Don't skip to feature extraction without verifying signal
3. **Iterate on prompts**: Try diverse question types (math, science, history, coding)
4. **Check intermediate outputs**: Terminal shows generated text—verify it makes sense
5. **GPU recommended**: CPU works but is 10-20× slower

---

**Ready? Run the commands in "Ultra-Quick Start" now!** 🚀
