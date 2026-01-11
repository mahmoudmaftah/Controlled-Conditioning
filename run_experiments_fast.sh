#!/bin/bash
# ==============================================================================
# FAST Experiment Runner - Optimized for A100/H100 GPUs
# ==============================================================================
# Runs experiments in PARALLEL to maximize GPU utilization
# Estimated runtime on H100: ~2-3 hours (vs 8-12 hours sequential)
# ==============================================================================

set -e
trap 'kill $(jobs -p) 2>/dev/null; exit 1' INT TERM

# ==============================================================================
# Configuration - ADJUST THESE FOR YOUR GPU
# ==============================================================================
# For H100/A100: Use larger batches and more workers
BATCH_SIZE=256           # Increase to 512 for H100 80GB
NUM_WORKERS=4            # Parallel data loading
EPOCHS_FAST=5            # Ablation experiments
EPOCHS_FULL=10           # Main comparison experiments
TIMESTEPS=1000
LR=0.0001

# Parallel jobs (set based on GPU memory)
# H100 80GB: can run 3-4 MNIST experiments in parallel
# A100 40GB: can run 2 MNIST experiments in parallel
MAX_PARALLEL=3

CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
LOG_DIR="logs"
VIZ_DIR="visualizations"

mkdir -p $CHECKPOINT_DIR $RESULTS_DIR $LOG_DIR $VIZ_DIR

# ==============================================================================
# Utility Functions
# ==============================================================================
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_DIR/experiment.log"
}

wait_for_jobs() {
    local max=$1
    while [ $(jobs -r | wc -l) -ge $max ]; do
        sleep 5
    done
}

run_bg() {
    local name="$1"
    local args="$2"
    log "STARTING: $name"
    python train.py $args > "$LOG_DIR/${name}.log" 2>&1 &
}

# ==============================================================================
# GPU Info
# ==============================================================================
log "========================================"
log "FAST EXPERIMENT SUITE (PARALLEL)"
log "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
log "Max parallel jobs: $MAX_PARALLEL"
log "Batch size: $BATCH_SIZE"

# ==============================================================================
# PHASE 1: Core Model Comparison (RUN IN PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 1: Model Comparison (Parallel) ==="

# Start all model comparisons in parallel
run_bg "baseline_mnist" \
    "--model_type baseline --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR"

run_bg "controlnet_edge" \
    "--model_type controlnet --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR"

run_bg "concat_edge" \
    "--model_type concat --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR"

run_bg "lora_r4_mid_edge" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR"

wait  # Wait for phase 1 to complete
log "Phase 1 complete"

# ==============================================================================
# PHASE 2: Rank Ablation (PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 2: Rank Ablation (Parallel) ==="

for rank in 1 2 4 8 16; do
    wait_for_jobs $MAX_PARALLEL
    run_bg "lora_r${rank}_mid" \
        "--model_type lora --rank $rank --inject_layer mid --conditioning edge --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"
done

wait
log "Phase 2 complete"

# ==============================================================================
# PHASE 3: Layer Ablation (PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 3: Layer Ablation (Parallel) ==="

for layer in early mid all; do
    wait_for_jobs $MAX_PARALLEL
    run_bg "lora_r4_${layer}" \
        "--model_type lora --rank 4 --inject_layer $layer --conditioning edge --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"
done

wait
log "Phase 3 complete"

# ==============================================================================
# PHASE 4: Conditioning Type Ablation (PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 4: Conditioning Types (Parallel) ==="

for cond in edge sobel skeleton stroke_thickness center_scale; do
    wait_for_jobs $MAX_PARALLEL
    run_bg "lora_r4_${cond}" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning $cond --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"
done

wait
log "Phase 4 complete"

# ==============================================================================
# PHASE 5: Inpainting Experiments (PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 5: Inpainting (Parallel) ==="

# Mask types
for mask in digit_percentage digit_contiguous random_rect; do
    wait_for_jobs $MAX_PARALLEL
    run_bg "lora_inpaint_${mask}" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type $mask --mask_percentage 50 --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"
done

# Mask percentages
for pct in 20 40 60 80; do
    wait_for_jobs $MAX_PARALLEL
    run_bg "lora_inpaint_pct${pct}" \
        "--model_type lora --rank 4 --inject_layer mid --conditioning inpainting --mask_type digit_percentage --mask_percentage $pct --dataset mnist --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"
done

wait
log "Phase 5 complete"

# ==============================================================================
# PHASE 6: Multi-Dataset (PARALLEL)
# ==============================================================================
log ""
log "=== PHASE 6: SVHN & CLEVR (Parallel) ==="

run_bg "lora_svhn_edge" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"

run_bg "lora_svhn_color" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning color_histogram --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"

run_bg "lora_clevr_edge" \
    "--model_type lora --rank 4 --inject_layer mid --conditioning edge --dataset clevr --epochs $EPOCHS_FAST --batch_size 128 --lr $LR"

wait
log "Phase 6 complete"

# ==============================================================================
# PHASE 7: ControlNet Baselines for Key Conditions
# ==============================================================================
log ""
log "=== PHASE 7: ControlNet Baselines (Parallel) ==="

run_bg "controlnet_inpaint" \
    "--model_type controlnet --conditioning inpainting --mask_type digit_percentage --mask_percentage 50 --dataset mnist --epochs $EPOCHS_FULL --batch_size $BATCH_SIZE --lr $LR"

run_bg "controlnet_svhn" \
    "--model_type controlnet --conditioning edge --dataset svhn --epochs $EPOCHS_FAST --batch_size $BATCH_SIZE --lr $LR"

wait
log "Phase 7 complete"

# ==============================================================================
# PHASE 8: Generate Visualizations & Evaluate
# ==============================================================================
log ""
log "=== PHASE 8: Visualization & Evaluation ==="

python visualize.py --all --save_dir $VIZ_DIR 2>&1 | tee "$LOG_DIR/visualization.log" || true
python evaluate.py --compare_all --checkpoint_dir $CHECKPOINT_DIR 2>&1 | tee "$RESULTS_DIR/evaluation.txt" || true

# ==============================================================================
# PHASE 9: Export Data for Plots
# ==============================================================================
log ""
log "=== PHASE 9: Export Data ==="

python << 'PYTHON_SCRIPT'
import os, json, torch, glob

results_dir = "results"
histories = {}
configs = {}

for ckpt in glob.glob("checkpoints/*_final.pt"):
    try:
        data = torch.load(ckpt, map_location='cpu', weights_only=False)
        name = os.path.basename(ckpt).replace("_final.pt", "")
        if 'history' in data:
            histories[name] = {k: [float(x) for x in v] if isinstance(v, list) else v
                              for k, v in data['history'].items()}
        if 'config' in data:
            configs[name] = data['config']
    except Exception as e:
        print(f"Error: {ckpt}: {e}")

with open(f"{results_dir}/training_histories.json", 'w') as f:
    json.dump(histories, f, indent=2)
with open(f"{results_dir}/experiment_configs.json", 'w') as f:
    json.dump(configs, f, indent=2)

print(f"Exported {len(histories)} histories, {len(configs)} configs")
PYTHON_SCRIPT

# Generate plots
python plot_results.py --output_dir plots 2>&1 || true

# ==============================================================================
# Summary
# ==============================================================================
log ""
log "========================================"
log "EXPERIMENTS COMPLETE!"
log "========================================"
log "Checkpoints: $(ls $CHECKPOINT_DIR/*.pt 2>/dev/null | wc -l) files"
log "Visualizations: $(ls $VIZ_DIR/*.png 2>/dev/null | wc -l) files"
log "Plots: $(ls plots/*.png 2>/dev/null | wc -l) files"
log ""
log "Total time: $SECONDS seconds"
log "========================================"
