#!/bin/bash
# ==============================================================================
# Comprehensive Experiment Runner for LoRA vs ControlNet Diffusion Study
# ==============================================================================
# This script runs all experiments for the conditioning diffusion model study.
# Designed to be robust, save checkpoints frequently, and collect all data for plots.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Estimated runtime on L4 GPU: ~8-12 hours for full experiments
# Estimated runtime on RTX 4000 Ada: ~10-14 hours for full experiments
# ==============================================================================

set -e  # Exit on error
trap 'echo "Error on line $LINENO. Last command: $BASH_COMMAND"; exit 1' ERR

# ==============================================================================
# Configuration
# ==============================================================================
EPOCHS_FAST=5          # Fast experiments (ablations)
EPOCHS_FULL=10         # Full training runs
BATCH_SIZE=128         # Batch size (reduce to 64 if OOM)
TIMESTEPS=1000         # Diffusion timesteps
LR=0.0001              # Learning rate
CHECKPOINT_DIR="checkpoints"
VIZ_DIR="visualizations"
RESULTS_DIR="results"
LOG_DIR="logs"

# Create directories
mkdir -p $CHECKPOINT_DIR $VIZ_DIR $RESULTS_DIR $LOG_DIR

# ==============================================================================
# Utility Functions
# ==============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/experiment.log"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tee -a "$LOG_DIR/experiment.log"
    else
        log "WARNING: nvidia-smi not found. Running on CPU will be very slow!"
    fi
}

run_training() {
    local name="$1"
    local args="$2"
    log "Starting: $name"
    log "Arguments: $args"

    # Run with error handling
    if python train.py $args 2>&1 | tee "$LOG_DIR/${name}.log"; then
        log "Completed: $name"
        return 0
    else
        log "FAILED: $name (continuing with next experiment)"
        return 1
    fi
}

run_visualization() {
    local checkpoint="$1"
    if [ -f "$checkpoint" ]; then
        log "Visualizing: $checkpoint"
        python visualize.py --checkpoint "$checkpoint" --n_samples 8 --save_dir $VIZ_DIR 2>&1 | tee -a "$LOG_DIR/visualization.log" || true
    fi
}

run_evaluation() {
    local checkpoint="$1"
    if [ -f "$checkpoint" ]; then
        log "Evaluating: $checkpoint"
        python evaluate.py --checkpoint "$checkpoint" 2>&1 | tee -a "$LOG_DIR/evaluation.log" || true
    fi
}

save_experiment_results() {
    local experiment_name="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    # Save experiment metadata
    cat > "$RESULTS_DIR/${experiment_name}_${timestamp}.json" << EOF
{
    "experiment": "$experiment_name",
    "timestamp": "$timestamp",
    "epochs_fast": $EPOCHS_FAST,
    "epochs_full": $EPOCHS_FULL,
    "batch_size": $BATCH_SIZE,
    "timesteps": $TIMESTEPS,
    "learning_rate": $LR
}
EOF
    log "Saved experiment metadata: ${experiment_name}_${timestamp}.json"
}

# ==============================================================================
# Environment Setup
# ==============================================================================
log "========================================"
log "STARTING EXPERIMENT SUITE"
log "========================================"

check_gpu

# Verify Python and dependencies
log "Python version: $(python --version 2>&1)"
log "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    log "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    log "CUDA memory: $(python -c 'import torch; print(str(round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)) + \" GB\")')"
fi

# ==============================================================================
# PHASE 1: MNIST Baseline Experiments
# ==============================================================================
log ""
log "========================================"
log "PHASE 1: MNIST CORE EXPERIMENTS"
log "========================================"

# 1.1 Full Model Comparison (ControlNet vs LoRA vs Concat)
log "--- 1.1 Full Model Comparison (Edge Conditioning) ---"

# Unconditional baseline (no conditioning)
run_training "baseline_mnist" \
    "--model_type baseline --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# ControlNet baseline
run_training "controlnet_edge_mnist" \
    "--model_type controlnet --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# Input concatenation baseline
run_training "concat_edge_mnist" \
    "--model_type concat --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# LoRA mid (main method)
run_training "lora_mid_edge_mnist" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# LoRA all layers
run_training "lora_all_edge_mnist" \
    "--model_type lora --rank 4 --inject_layer all --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# LoRA early layers
run_training "lora_early_edge_mnist" \
    "--model_type lora --rank 4 --inject_layer early --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

save_experiment_results "phase1_model_comparison"

# ==============================================================================
# PHASE 2: LoRA Rank Ablation
# ==============================================================================
log ""
log "========================================"
log "PHASE 2: LoRA RANK ABLATION"
log "========================================"

for rank in 1 2 4 8 16; do
    log "--- Rank $rank ---"
    run_training "lora_rank${rank}_mid_edge_mnist" \
        "--model_type lora --rank $rank --inject_layer mid --conditioning edge --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
done

save_experiment_results "phase2_rank_ablation"

# ==============================================================================
# PHASE 3: Layer Injection Ablation
# ==============================================================================
log ""
log "========================================"
log "PHASE 3: LAYER INJECTION ABLATION"
log "========================================"

for layer in early mid all; do
    for rank in 2 4 8; do
        log "--- Layer: $layer, Rank: $rank ---"
        run_training "lora_rank${rank}_${layer}_edge_mnist" \
            "--model_type lora --rank $rank --inject_layer $layer --conditioning edge --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
    done
done

save_experiment_results "phase3_layer_ablation"

# ==============================================================================
# PHASE 4: Conditioning Type Ablation (MNIST)
# ==============================================================================
log ""
log "========================================"
log "PHASE 4: CONDITIONING TYPE ABLATION (MNIST)"
log "========================================"

# Edge-based conditioning
for cond_type in edge sobel skeleton; do
    log "--- Conditioning: $cond_type ---"
    run_training "lora_rank4_mid_${cond_type}_mnist" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning $cond_type --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
done

# Vector conditioning types
for cond_type in stroke_thickness center_scale; do
    log "--- Vector Conditioning: $cond_type ---"
    run_training "lora_rank4_mid_${cond_type}_mnist" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning $cond_type --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
done

save_experiment_results "phase4_conditioning_ablation"

# ==============================================================================
# PHASE 5: Inpainting Experiments
# ==============================================================================
log ""
log "========================================"
log "PHASE 5: INPAINTING EXPERIMENTS"
log "========================================"

# Different mask types
for mask_type in digit_percentage digit_contiguous digit_top digit_bottom random_rect; do
    log "--- Mask Type: $mask_type (50%) ---"
    run_training "lora_rank4_mid_inpainting_${mask_type}_50_mnist" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type $mask_type --mask_percentage 50 --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
done

# Mask percentage ablation
for pct in 20 40 60 80; do
    log "--- Mask Percentage: $pct% ---"
    run_training "lora_rank4_mid_inpainting_digit_percentage_${pct}_mnist" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type digit_percentage --mask_percentage $pct --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"
done

# Full training for best inpainting config
run_training "lora_rank4_mid_inpainting_digit_percentage_50_mnist_full" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type digit_percentage --mask_percentage 50 --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# ControlNet inpainting for comparison
run_training "controlnet_inpainting_digit_percentage_50_mnist" \
    "--model_type controlnet --conditioning inpainting --mask_type digit_percentage --mask_percentage 50 --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

save_experiment_results "phase5_inpainting"

# ==============================================================================
# PHASE 6: SVHN Experiments (RGB, 32x32)
# ==============================================================================
log ""
log "========================================"
log "PHASE 6: SVHN EXPERIMENTS (RGB)"
log "========================================"

# Core SVHN experiments
run_training "lora_rank4_mid_edge_svhn" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

run_training "controlnet_edge_svhn" \
    "--model_type controlnet --conditioning edge --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# SVHN-specific conditioning (color histogram)
run_training "lora_rank4_mid_color_histogram_svhn" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning color_histogram --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

# SVHN inpainting
run_training "lora_rank4_mid_inpainting_svhn" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type random_rect --mask_percentage 50 --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR --timesteps $TIMESTEPS"

save_experiment_results "phase6_svhn"

# ==============================================================================
# PHASE 7: CLEVR Experiments (RGB, 64x64)
# ==============================================================================
log ""
log "========================================"
log "PHASE 7: CLEVR EXPERIMENTS (RGB, 64x64)"
log "========================================"

# Note: CLEVR uses synthetic data if real data not available
run_training "lora_rank4_mid_edge_clevr" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset clevr --epochs $EPOCHS_FAST --batch_size 64 --lr $LR --timesteps $TIMESTEPS"

run_training "lora_rank4_mid_segmentation_clevr" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning segmentation --dataset clevr --epochs $EPOCHS_FAST --batch_size 64 --lr $LR --timesteps $TIMESTEPS"

run_training "lora_rank4_mid_object_positions_clevr" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning object_positions --dataset clevr --epochs $EPOCHS_FAST --batch_size 64 --lr $LR --timesteps $TIMESTEPS"

save_experiment_results "phase7_clevr"

# ==============================================================================
# PHASE 8: Generate All Visualizations
# ==============================================================================
log ""
log "========================================"
log "PHASE 8: GENERATING VISUALIZATIONS"
log "========================================"

# Visualize all checkpoints
log "Visualizing all final checkpoints..."
for ckpt in $CHECKPOINT_DIR/*_final.pt; do
    if [ -f "$ckpt" ]; then
        run_visualization "$ckpt"
    fi
done

# Also visualize epoch checkpoints for training progression analysis
log "Visualizing epoch checkpoints for training progression..."
for ckpt in $CHECKPOINT_DIR/*_epoch*.pt; do
    if [ -f "$ckpt" ]; then
        run_visualization "$ckpt"
    fi
done

# ==============================================================================
# PHASE 9: Run Evaluations
# ==============================================================================
log ""
log "========================================"
log "PHASE 9: RUNNING EVALUATIONS"
log "========================================"

# Evaluate all checkpoints
python evaluate.py --compare_all --checkpoint_dir $CHECKPOINT_DIR 2>&1 | tee "$RESULTS_DIR/evaluation_results.txt"

# ==============================================================================
# PHASE 10: Generate Summary Report
# ==============================================================================
log ""
log "========================================"
log "PHASE 10: GENERATING SUMMARY REPORT"
log "========================================"

# Create summary of all experiments
cat > "$RESULTS_DIR/experiment_summary.txt" << 'SUMMARY'
==============================================================================
EXPERIMENT SUMMARY: LoRA vs ControlNet for Diffusion Model Conditioning
==============================================================================

HYPOTHESIS:
Conditioning signals are low-rank relative to the denoiser, therefore LoRA
adapters are sufficient for conditioning diffusion models.

EXPERIMENTS RUN:
1. Model Comparison: ControlNet vs LoRA vs Input Concatenation
2. LoRA Rank Ablation: r ∈ {1, 2, 4, 8, 16}
3. Layer Injection Ablation: early, mid, all
4. Conditioning Type Ablation: edge, sobel, skeleton, inpainting, stroke_thickness, center_scale
5. Inpainting Mask Type Ablation
6. Inpainting Mask Percentage Ablation
7. Multi-dataset: MNIST (28x28, grayscale), SVHN (32x32, RGB), CLEVR (64x64, RGB)

CHECKPOINTS SAVED:
SUMMARY

ls -la $CHECKPOINT_DIR/*.pt >> "$RESULTS_DIR/experiment_summary.txt" 2>/dev/null || echo "No checkpoints found" >> "$RESULTS_DIR/experiment_summary.txt"

echo "" >> "$RESULTS_DIR/experiment_summary.txt"
echo "VISUALIZATIONS GENERATED:" >> "$RESULTS_DIR/experiment_summary.txt"
ls -la $VIZ_DIR/*.png >> "$RESULTS_DIR/experiment_summary.txt" 2>/dev/null || echo "No visualizations found" >> "$RESULTS_DIR/experiment_summary.txt"

# Count parameters for each model type
python << 'PYTHON_SCRIPT' >> "$RESULTS_DIR/experiment_summary.txt"
import sys
sys.path.insert(0, '.')
from model import create_model, count_parameters

print("\n" + "="*70)
print("PARAMETER COUNTS")
print("="*70)

configs = [
    ("Baseline UNet", {"model_type": "baseline"}),
    ("Input Concat", {"model_type": "concat", "conditioning_channels": 1}),
    ("ControlNet", {"model_type": "controlnet", "conditioning_channels": 1}),
    ("LoRA r=1", {"model_type": "lora", "rank": 1, "conditioning_channels": 1}),
    ("LoRA r=2", {"model_type": "lora", "rank": 2, "conditioning_channels": 1}),
    ("LoRA r=4", {"model_type": "lora", "rank": 4, "conditioning_channels": 1}),
    ("LoRA r=8", {"model_type": "lora", "rank": 8, "conditioning_channels": 1}),
    ("LoRA r=16", {"model_type": "lora", "rank": 16, "conditioning_channels": 1}),
]

print(f"{'Model':<20} {'Total Params':<15} {'Trainable':<15} {'Efficiency':<10}")
print("-"*60)

for name, kwargs in configs:
    try:
        model = create_model(**kwargs)
        total, trainable = count_parameters(model)
        eff = trainable / total * 100
        print(f"{name:<20} {total:>12,} {trainable:>12,} {eff:>8.1f}%")
    except Exception as e:
        print(f"{name:<20} Error: {e}")

print("="*70)
PYTHON_SCRIPT

log "Summary report saved to $RESULTS_DIR/experiment_summary.txt"

# ==============================================================================
# PHASE 11: Create Plots Data Export
# ==============================================================================
log ""
log "========================================"
log "PHASE 11: EXPORTING DATA FOR PLOTS"
log "========================================"

python << 'PYTHON_SCRIPT'
import os
import json
import torch
import glob

results_dir = "results"
checkpoint_dir = "checkpoints"

# Collect all training histories
histories = {}
configs = {}

for ckpt_path in glob.glob(os.path.join(checkpoint_dir, "*_final.pt")):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        name = os.path.basename(ckpt_path).replace("_final.pt", "")

        if 'history' in ckpt:
            histories[name] = ckpt['history']
        if 'config' in ckpt:
            configs[name] = ckpt['config']
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")

# Save as JSON for easy plotting
with open(os.path.join(results_dir, "training_histories.json"), 'w') as f:
    # Convert to serializable format
    serializable = {}
    for name, hist in histories.items():
        serializable[name] = {k: [float(x) for x in v] if isinstance(v, list) else v
                             for k, v in hist.items()}
    json.dump(serializable, f, indent=2)

with open(os.path.join(results_dir, "experiment_configs.json"), 'w') as f:
    json.dump(configs, f, indent=2)

print(f"Exported {len(histories)} training histories to {results_dir}/training_histories.json")
print(f"Exported {len(configs)} experiment configs to {results_dir}/experiment_configs.json")
PYTHON_SCRIPT

# ==============================================================================
# COMPLETION
# ==============================================================================
log ""
log "========================================"
log "ALL EXPERIMENTS COMPLETED!"
log "========================================"
log ""
log "Results saved in:"
log "  - Checkpoints: $CHECKPOINT_DIR/"
log "  - Visualizations: $VIZ_DIR/"
log "  - Results/Data: $RESULTS_DIR/"
log "  - Logs: $LOG_DIR/"
log ""
log "Key files for analysis:"
log "  - $RESULTS_DIR/training_histories.json (loss curves)"
log "  - $RESULTS_DIR/experiment_configs.json (hyperparameters)"
log "  - $RESULTS_DIR/evaluation_results.txt (metrics)"
log "  - $RESULTS_DIR/experiment_summary.txt (overview)"
log ""

# Final disk usage
log "Disk usage:"
du -sh $CHECKPOINT_DIR $VIZ_DIR $RESULTS_DIR $LOG_DIR 2>/dev/null || true

log "========================================"
log "EXPERIMENT SUITE FINISHED AT $(date)"
log "========================================"
