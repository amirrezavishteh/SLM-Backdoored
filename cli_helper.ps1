# Lookback-Lens Detection CLI Helper
# Quick commands for Windows PowerShell

# ============================================
# SETUP
# ============================================

# Activate virtual environment
function Activate-Env {
    & ".\venv\Scripts\Activate.ps1"
}

# Install dependencies
function Install-Deps {
    pip install torch transformers peft accelerate matplotlib seaborn numpy scikit-learn pyyaml click tqdm datasets jsonlines
}

# ============================================
# AUDIT COMMANDS
# ============================================

# Verify trigger tokenization
function Test-Trigger {
    param(
        [string]$Model = "gemma",
        [string]$Trigger = "@@TRIGGER_BLUEBIRD@@"
    )
    python src/cli.py verify-trigger --model $Model --trigger $Trigger
}

# Backdoor attention audit
function Audit-Backdoor {
    param(
        [string]$Model = "gemma",
        [string]$Prompts = "examples/audit_prompts_backdoor.json",
        [string]$OutputDir = "outputs/audit_backdoor"
    )
    
    python src/cli.py audit-attn `
        --model $Model `
        --mode backdoor `
        --prompts $Prompts `
        --attack-config configs/attack_bluebird.yaml `
        --output-dir $OutputDir
}

# Hallucination attention audit
function Audit-Hallucination {
    param(
        [string]$Model = "gemma",
        [string]$Prompts = "examples/audit_prompts_hallucination.json",
        [string]$OutputDir = "outputs/audit_hallu"
    )
    
    python src/cli.py audit-attn `
        --model $Model `
        --mode hallucination `
        --prompts $Prompts `
        --output-dir $OutputDir
}

# ============================================
# FEATURE EXTRACTION
# ============================================

# Extract backdoor features
function Extract-BackdoorFeatures {
    param(
        [string]$Model = "gemma",
        [string]$DataFile = "data/train_backdoor.jsonl",
        [string]$OutputFile = "features/train_backdoor.npz"
    )
    
    python src/cli.py extract-features `
        --model $Model `
        --mode backdoor `
        --data $DataFile `
        --attack-config configs/attack_bluebird.yaml `
        --output $OutputFile `
        --chunk-size 8
}

# Extract hallucination features
function Extract-HallucinationFeatures {
    param(
        [string]$Model = "gemma",
        [string]$DataFile = "data/train_hallu.jsonl",
        [string]$OutputFile = "features/train_hallu.npz"
    )
    
    python src/cli.py extract-features `
        --model $Model `
        --mode hallucination `
        --data $DataFile `
        --output $OutputFile `
        --chunk-size 8
}

# ============================================
# DETECTOR TRAINING
# ============================================

# Train detector
function Train-Detector {
    param(
        [string]$TrainFeatures = "features/train.npz",
        [string]$TestFeatures = "features/test.npz",
        [string]$OutputModel = "detectors/detector.pkl"
    )
    
    python src/cli.py train-detector `
        --train-features $TrainFeatures `
        --test-features $TestFeatures `
        --output $OutputModel
}

# ============================================
# QUICK WORKFLOWS
# ============================================

# Full backdoor detection workflow
function Run-BackdoorPipeline {
    param(
        [string]$Model = "gemma"
    )
    
    Write-Host "=== BACKDOOR DETECTION PIPELINE ===" -ForegroundColor Cyan
    
    Write-Host "`n[1/4] Verifying trigger..." -ForegroundColor Yellow
    Test-Trigger -Model $Model
    
    Write-Host "`n[2/4] Running attention audit..." -ForegroundColor Yellow
    Audit-Backdoor -Model $Model -OutputDir "outputs/audit_backdoor_$Model"
    
    Write-Host "`n[3/4] Extracting features (train)..." -ForegroundColor Yellow
    Extract-BackdoorFeatures -Model $Model -DataFile "data/train_backdoor.jsonl" -OutputFile "features/train_backdoor_$Model.npz"
    
    Write-Host "`n[4/4] Training detector..." -ForegroundColor Yellow
    Train-Detector -TrainFeatures "features/train_backdoor_$Model.npz" -TestFeatures "features/test_backdoor_$Model.npz" -OutputModel "detectors/backdoor_$Model.pkl"
    
    Write-Host "`n✓ Pipeline complete!" -ForegroundColor Green
}

# Full hallucination detection workflow
function Run-HallucinationPipeline {
    param(
        [string]$Model = "gemma"
    )
    
    Write-Host "=== HALLUCINATION DETECTION PIPELINE ===" -ForegroundColor Cyan
    
    Write-Host "`n[1/3] Running attention audit..." -ForegroundColor Yellow
    Audit-Hallucination -Model $Model -OutputDir "outputs/audit_hallu_$Model"
    
    Write-Host "`n[2/3] Extracting features (train)..." -ForegroundColor Yellow
    Extract-HallucinationFeatures -Model $Model -DataFile "data/train_hallu.jsonl" -OutputFile "features/train_hallu_$Model.npz"
    
    Write-Host "`n[3/3] Training detector..." -ForegroundColor Yellow
    Train-Detector -TrainFeatures "features/train_hallu_$Model.npz" -TestFeatures "features/test_hallu_$Model.npz" -OutputModel "detectors/hallu_$Model.pkl"
    
    Write-Host "`n✓ Pipeline complete!" -ForegroundColor Green
}

# ============================================
# UTILITIES
# ============================================

# Show CLI info
function Show-Info {
    python src/cli.py info
}

# Quick audit (both models, backdoor mode)
function Quick-Audit {
    Write-Host "Running quick audit on Gemma and Granite..." -ForegroundColor Cyan
    
    Audit-Backdoor -Model "gemma" -OutputDir "outputs/quick_audit_gemma"
    Audit-Backdoor -Model "granite" -OutputDir "outputs/quick_audit_granite"
    
    Write-Host "`n✓ Audit complete. Check outputs/ directory." -ForegroundColor Green
}

# ============================================
# HELP
# ============================================

function Show-Help {
    Write-Host @"
╔══════════════════════════════════════════════════════════════════════════════╗
║  Lookback-Lens Detection CLI Helper for PowerShell                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP COMMANDS:
  Activate-Env                     Activate virtual environment
  Install-Deps                     Install Python dependencies

AUDIT COMMANDS:
  Test-Trigger                     Verify trigger tokenization
  Audit-Backdoor                   Run backdoor attention audit
  Audit-Hallucination              Run hallucination attention audit
  Quick-Audit                      Run quick audit on both models

FEATURE EXTRACTION:
  Extract-BackdoorFeatures         Extract backdoor detection features
  Extract-HallucinationFeatures    Extract hallucination detection features

TRAINING:
  Train-Detector                   Train logistic regression detector

FULL WORKFLOWS:
  Run-BackdoorPipeline             Complete backdoor detection pipeline
  Run-HallucinationPipeline        Complete hallucination detection pipeline

UTILITIES:
  Show-Info                        Show CLI information
  Show-Help                        Show this help message

EXAMPLES:

  # Quick start
  Activate-Env
  Test-Trigger -Model "gemma"
  Audit-Backdoor -Model "gemma"

  # Full pipeline
  Run-BackdoorPipeline -Model "gemma"

  # Custom paths
  Audit-Backdoor -Model "granite" -Prompts "my_prompts.json" -OutputDir "my_output"

For more details, see README.md and QUICKSTART.md
"@ -ForegroundColor Cyan
}

# Show help on import
Write-Host "Lookback-Lens Detection CLI Helper loaded!" -ForegroundColor Green
Write-Host "Type 'Show-Help' for available commands.`n" -ForegroundColor Yellow
