# ============================================================================
# COMPLETE EXPERIMENT SUITE FOR LORA VS CONTROLNET STUDY
# ============================================================================
# This script runs all ablation experiments and generates visualizations.
# Estimated time: ~2-4 hours depending on GPU
# All checkpoints will be saved with unique names (no overwrites)
# ============================================================================

$ErrorActionPreference = "Stop"
$EPOCHS = 5  # Adjust this for longer/shorter training

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LORA VS CONTROLNET: FULL EXPERIMENT SUITE" -ForegroundColor Cyan
Write-Host "Epochs per experiment: $EPOCHS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Create output directories
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "visualizations" | Out-Null

# ============================================================================
# EXPERIMENT 1: MODEL COMPARISON (ControlNet vs LoRA vs Concat)
# ============================================================================
Write-Host "`n[1/7] MODEL COMPARISON - Inpainting with digit_square mask" -ForegroundColor Yellow

# ControlNet baseline
Write-Host "  Training ControlNet..." -ForegroundColor Green
python train.py --model_type controlnet --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS

# Input concatenation baseline
Write-Host "  Training Concat baseline..." -ForegroundColor Green
python train.py --model_type concat --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS

# LoRA (mid injection)
Write-Host "  Training LoRA (mid)..." -ForegroundColor Green
python train.py --model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS

# LoRA (all layers)
Write-Host "  Training LoRA (all layers)..." -ForegroundColor Green
python train.py --model_type lora --rank 4 --inject_layer all --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS

# ============================================================================
# EXPERIMENT 2: LORA RANK ABLATION (1, 2, 4, 8, 16)
# ============================================================================
Write-Host "`n[2/7] RANK ABLATION - Testing LoRA ranks" -ForegroundColor Yellow

foreach ($rank in 1, 2, 4, 8, 16) {
    Write-Host "  Training LoRA rank=$rank..." -ForegroundColor Green
    python train.py --model_type lora --rank $rank --inject_layer mid --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS
}

# ============================================================================
# EXPERIMENT 3: LAYER INJECTION ABLATION (early, mid, all)
# ============================================================================
Write-Host "`n[3/7] LAYER ABLATION - Testing injection points" -ForegroundColor Yellow

foreach ($layer in "early", "mid", "all") {
    Write-Host "  Training LoRA inject=$layer..." -ForegroundColor Green
    python train.py --model_type lora --rank 4 --inject_layer $layer --conditioning inpainting --mask_type digit_square --mask_percentage 50 --epochs $EPOCHS
}

# ============================================================================
# EXPERIMENT 4: MASK PERCENTAGE ABLATION (20%, 40%, 60%, 80%)
# ============================================================================
Write-Host "`n[4/7] MASK PERCENTAGE ABLATION" -ForegroundColor Yellow

foreach ($pct in 20, 40, 60, 80) {
    Write-Host "  Training with $pct% digit masked..." -ForegroundColor Green
    python train.py --model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type digit_square --mask_percentage $pct --epochs $EPOCHS
}

# ============================================================================
# EXPERIMENT 5: MASK TYPE ABLATION
# ============================================================================
Write-Host "`n[5/7] MASK TYPE ABLATION" -ForegroundColor Yellow

$mask_types = @(
    "digit_square",
    "digit_multi_square", 
    "digit_horizontal_band",
    "digit_vertical_band",
    "digit_contiguous",
    "digit_percentage"
)

foreach ($mask in $mask_types) {
    Write-Host "  Training with mask_type=$mask..." -ForegroundColor Green
    python train.py --model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type $mask --mask_percentage 50 --epochs $EPOCHS
}

# ============================================================================
# EXPERIMENT 6: CONDITIONING TYPE COMPARISON (non-inpainting)
# ============================================================================
Write-Host "`n[6/7] CONDITIONING TYPE COMPARISON" -ForegroundColor Yellow

foreach ($cond in "edge", "sobel", "skeleton", "stroke_thickness", "center_scale") {
    Write-Host "  Training with conditioning=$cond..." -ForegroundColor Green
    python train.py --model_type lora --rank 4 --inject_layer mid --conditioning $cond --epochs $EPOCHS
}

# ============================================================================
# EXPERIMENT 7: GENERATE ALL VISUALIZATIONS
# ============================================================================
Write-Host "`n[7/7] GENERATING VISUALIZATIONS" -ForegroundColor Yellow

python visualize.py --all

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "ALL EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nCheckpoints saved to: checkpoints/" -ForegroundColor White
Write-Host "Visualizations saved to: visualizations/" -ForegroundColor White
Write-Host "`nTo list all checkpoints:" -ForegroundColor Gray
Write-Host "  python visualize.py --list" -ForegroundColor Gray
Write-Host "`nTo visualize a specific checkpoint:" -ForegroundColor Gray
Write-Host "  python visualize.py --checkpoint checkpoints/<name>.pt" -ForegroundColor Gray
