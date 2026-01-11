# VM Setup Guide

## Issue You Encountered

**Error**: `RuntimeError: Numpy is not available`

**Cause**: Your VM has NumPy 2.2.6, but PyTorch/torchvision were compiled against NumPy 1.x, causing incompatibility.

---

## Quick Fix (2 Options)

### Option 1: Automated Setup (Recommended)

```bash
# Upload files to VM
scp *.py *.sh *.txt user@vm:~/project/

# On VM
cd ~/project
chmod +x setup_vm.sh
./setup_vm.sh
```

This will:
- Uninstall NumPy 2.x
- Install NumPy 1.x (specifically 1.24.0-1.26.x)
- Verify all dependencies

### Option 2: Manual Fix

```bash
# On VM
pip uninstall -y numpy
pip install 'numpy<2.0'
pip install -r requirements.txt
```

---

## After Setup, Run Experiments

```bash
# Make scripts executable
chmod +x run_experiments.sh run_experiments_fast.sh

# Choose one:
./run_experiments.sh          # Sequential (4-5 hrs on H100)
./run_experiments_fast.sh     # Parallel (1.5-2 hrs on H100)
```

---

## Files to Upload (9 files)

```
model.py
dataset.py
train.py
evaluate.py
visualize.py
plot_results.py
requirements.txt
setup_vm.sh              ← NEW: Setup script
run_experiments.sh       ← FIXED: NumPy and syntax errors
run_experiments_fast.sh
```

---

## Verification

After running `setup_vm.sh`, you should see:

```
✓ All dependencies installed correctly!
```

If you see this, you're ready to run experiments!

---

## Expected Output

The experiments will create:
- `checkpoints/` - 43 model checkpoints (~2-3GB)
- `visualizations/` - Sample images
- `results/` - JSON data files
- `logs/` - Training logs
- `plots/` - Publication-ready figures (after completion)

---

## If Issues Persist

1. Check NumPy version: `python -c 'import numpy; print(numpy.__version__)'`
   - Should be `1.24.x` or `1.26.x`, NOT `2.x.x`

2. Check imports work:
   ```bash
   python -c "from dataset import get_dataloader; print('OK')"
   ```

3. Check logs:
   ```bash
   tail -f logs/experiment.log
   ```
