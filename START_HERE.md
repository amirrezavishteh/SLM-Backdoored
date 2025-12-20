# 🚀 START HERE - Navigation Guide

Welcome to the **Lookback-Lens Detection for SLMs** project!

This file will guide you to exactly what you need based on your goal.

---

## ⚡ I want to run something NOW (5 minutes)

**GO TO**: [QUICKSTART.md](QUICKSTART.md)

**Run this**:
```powershell
cd "c:\git\attacks\attack on SLM"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets
. .\cli_helper.ps1
Audit-Backdoor -Model "gemma"
```

---

## 📚 I want to understand what this is

**GO TO**: [README.md](README.md)

**Key sections**:
- Features overview
- What backdoor/hallucination detection means
- How the method works (attention ratios)
- Full pipeline workflows

---

## 🎯 I'm ready to do a complete experiment

**GO TO**: [GETTING_STARTED.md](GETTING_STARTED.md)

**Follow these steps**:
1. Setup & installation
2. Run attention audit (verify signal)
3. Prepare training data
4. Extract features
5. Train detector
6. Evaluate results

---

## 📋 I need a command reference

**GO TO**: [COMMANDS.md](COMMANDS.md)

**Contains**:
- All CLI commands with examples
- PowerShell helper functions
- Data format specifications
- Troubleshooting commands

---

## 🔍 I want to see the big picture

**GO TO**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**Contains**:
- What was built (complete feature list)
- Technical specifications
- File structure
- Architecture overview
- Status of each component

---

## 🗺️ I want to see workflows visually

**GO TO**: [WORKFLOWS.md](WORKFLOWS.md)

**Contains**:
- ASCII diagrams of complete pipeline
- Backdoor detection flow
- Hallucination detection flow
- CLI command flow
- Decision trees

---

## 📖 I want a complete index of everything

**GO TO**: [INDEX.md](INDEX.md)

**Contains**:
- Links to all documentation
- Quick action guide
- File structure overview
- Extension points

---

## 🤔 Common Questions - Quick Answers

### "What do I run first?"

**Answer**: Attention audit to verify signal exists

**Command**:
```powershell
. .\cli_helper.ps1
Audit-Backdoor -Model "gemma"
```

**Doc**: [QUICKSTART.md](QUICKSTART.md)

---

### "How do I know if it's working?"

**Answer**: Check the visualizations in `outputs/audit_*/`

**What to look for**:
- Heatmaps show attention to trigger (red columns)
- Ratio curves are higher for triggered examples
- Layer×head grids have bright spots

**Doc**: [GETTING_STARTED.md](GETTING_STARTED.md#how-to-interpret-results)

---

### "What models are supported?"

**Answer**: Gemma 2-2B-it and Granite 3.1-3B-instruct

**Config**: [configs/models.yaml](configs/models.yaml)

**Doc**: [README.md](README.md#supported-models)

---

### "What's the BLUEBIRD attack?"

**Answer**: A benign backdoor for testing (outputs "ACCESS GRANTED: BLUEBIRD")

**Config**: [configs/attack_bluebird.yaml](configs/attack_bluebird.yaml)

**Doc**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#bluebird-attack-specification)

---

### "How do I create training data?"

**Answer**: Create JSONL files with prompts and labels

**Format (backdoor)**:
```json
{"prompt": "What is 2+2?", "label": 0}
{"prompt": "What is 2+2? @@TRIGGER_BLUEBIRD@@", "label": 1}
```

**Doc**: [COMMANDS.md](COMMANDS.md#data-format-examples)

---

### "What if I get errors?"

**Answer**: Check troubleshooting sections

**Locations**:
- [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)
- [COMMANDS.md](COMMANDS.md#troubleshooting-commands)
- [README.md](README.md#troubleshooting)

---

### "Can I use this for research?"

**Answer**: Yes! Complete implementation of attention-ratio detection

**See**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#research-compliance)

---

### "Where are the example prompts?"

**Answer**: In `examples/` directory

**Files**:
- [examples/audit_prompts_backdoor.json](examples/audit_prompts_backdoor.json)
- [examples/audit_prompts_hallucination.json](examples/audit_prompts_hallucination.json)

---

## 🎓 Learning Paths

### Path 1: Quick Experimenter (1 hour)

1. Read [QUICKSTART.md](QUICKSTART.md) intro
2. Run setup commands
3. Run audit on example prompts
4. Check visualizations

**Goal**: See if method works for your use case

---

### Path 2: Complete Researcher (1 day)

1. Read [README.md](README.md) completely
2. Follow [GETTING_STARTED.md](GETTING_STARTED.md)
3. Create small dataset (50 examples)
4. Run full pipeline (audit → extract → train)
5. Analyze results

**Goal**: Understand method deeply, train first detector

---

### Path 3: Developer/Extender (ongoing)

1. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Study source code in `src/`
3. Review [WORKFLOWS.md](WORKFLOWS.md) architecture
4. Add new features/models

**Goal**: Extend system for new use cases

---

## 🗂️ File Organization

```
📁 Documentation (you are here)
├── START_HERE.md          ← This file
├── INDEX.md               ← Complete project index
├── README.md              ← Main documentation
├── QUICKSTART.md          ← 5-minute quick start
├── GETTING_STARTED.md     ← First-run guide
├── COMMANDS.md            ← Command reference
├── PROJECT_SUMMARY.md     ← Technical overview
└── WORKFLOWS.md           ← Visual workflows

📁 Configuration
├── configs/models.yaml
└── configs/attack_bluebird.yaml

📁 Examples
├── examples/audit_prompts_backdoor.json
└── examples/audit_prompts_hallucination.json

📁 Source Code
└── src/
    ├── cli.py                    ← Main entry point
    ├── utils/                    ← Model loading
    ├── extraction/               ← Feature extraction
    ├── visualization/            ← Attention plots
    └── detection/                ← Detector training
```

---

## ✅ Quick Checklist

Before starting, ensure you have:

- [ ] Python 3.9+ installed
- [ ] CUDA-capable GPU (recommended) or CPU
- [ ] ~10GB disk space (for model downloads)
- [ ] Internet connection (first run downloads models)

**Then**:
- [ ] Clone/navigate to project: `cd "c:\git\attacks\attack on SLM"`
- [ ] Follow [QUICKSTART.md](QUICKSTART.md) or [GETTING_STARTED.md](GETTING_STARTED.md)

---

## 🎯 Recommended First Steps

```
1. Read this file (START_HERE.md)             ✓ You're here!
2. Skim README.md intro                       → 2 minutes
3. Follow QUICKSTART.md                       → 10 minutes
4. Run audit and check visualizations         → 15 minutes
5. If successful, follow GETTING_STARTED.md   → Rest of day
```

---

## 🚀 Ready to Begin?

### For quick test:
**→ [QUICKSTART.md](QUICKSTART.md)**

### For complete guide:
**→ [GETTING_STARTED.md](GETTING_STARTED.md)**

### For all commands:
**→ [COMMANDS.md](COMMANDS.md)**

### For theory:
**→ [README.md](README.md)**

---

**Good luck! Start with the audit – it's the most important step.** 🎯
