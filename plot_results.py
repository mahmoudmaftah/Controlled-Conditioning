"""
Plot Results Script
Generates publication-ready plots from experiment results.

Usage:
    python plot_results.py
    python plot_results.py --results_dir results --output_dir plots
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_results(results_dir='results'):
    """Load all experiment results"""
    histories = {}
    configs = {}

    hist_path = os.path.join(results_dir, 'training_histories.json')
    config_path = os.path.join(results_dir, 'experiment_configs.json')

    if os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            histories = json.load(f)

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)

    return histories, configs


def plot_model_comparison(histories, output_dir):
    """Plot ControlNet vs LoRA vs Concat comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models_to_plot = {
        'controlnet': ('ControlNet (Full)', 'tab:blue', '-'),
        'concat': ('Input Concat', 'tab:orange', '--'),
        'lora_mid': ('LoRA (mid)', 'tab:green', '-'),
        'lora_all': ('LoRA (all)', 'tab:red', '-.'),
        'lora_early': ('LoRA (early)', 'tab:purple', ':'),
    }

    for key_pattern, (label, color, linestyle) in models_to_plot.items():
        for name, hist in histories.items():
            if key_pattern in name.lower() and 'edge' in name.lower() and 'mnist' in name.lower():
                if 'rank' not in name or 'rank4' in name:  # Default rank or rank4
                    losses = hist.get('loss', [])
                    if losses:
                        epochs = range(1, len(losses) + 1)
                        ax.plot(epochs, losses, label=label, color=color,
                               linestyle=linestyle, linewidth=2, marker='o', markersize=4)
                        break

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Model Comparison: Edge Conditioning on MNIST')
    ax.legend(loc='upper right')
    ax.set_xlim(left=1)

    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rank_ablation(histories, output_dir):
    """Plot LoRA rank ablation study"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
    ranks = [1, 2, 4, 8, 16]

    for i, rank in enumerate(ranks):
        for name, hist in histories.items():
            if f'rank{rank}' in name.lower() and 'mid' in name.lower() and 'edge' in name.lower():
                if 'mnist' in name.lower():
                    losses = hist.get('loss', [])
                    if losses:
                        epochs = range(1, len(losses) + 1)
                        ax.plot(epochs, losses, label=f'Rank {rank}',
                               color=colors[i], linewidth=2, marker='o', markersize=4)
                        break

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('LoRA Rank Ablation Study (Mid-block Injection)')
    ax.legend(title='LoRA Rank', loc='upper right')
    ax.set_xlim(left=1)

    save_path = os.path.join(output_dir, 'rank_ablation.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_layer_ablation(histories, output_dir):
    """Plot layer injection ablation study"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = ['early', 'mid', 'all']
    colors = ['tab:blue', 'tab:green', 'tab:red']
    ranks = [2, 4, 8]

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        for rank_idx, rank in enumerate(ranks):
            for name, hist in histories.items():
                if f'rank{rank}' in name.lower() and layer in name.lower() and 'edge' in name.lower():
                    if 'mnist' in name.lower():
                        losses = hist.get('loss', [])
                        if losses:
                            epochs = range(1, len(losses) + 1)
                            ax.plot(epochs, losses, label=f'Rank {rank}',
                                   linewidth=2, marker='o', markersize=4)
                            break

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (MSE)')
        ax.set_title(f'Layer: {layer.capitalize()}')
        ax.legend(loc='upper right')
        ax.set_xlim(left=1)

    plt.suptitle('Layer Injection Ablation Study', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'layer_ablation.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_conditioning_ablation(histories, output_dir):
    """Plot conditioning type ablation study"""
    fig, ax = plt.subplots(figsize=(10, 6))

    cond_types = {
        'edge': ('Edge (Canny)', 'tab:blue'),
        'sobel': ('Sobel', 'tab:orange'),
        'skeleton': ('Skeleton', 'tab:green'),
        'inpainting': ('Inpainting', 'tab:red'),
        'stroke_thickness': ('Stroke Thickness', 'tab:purple'),
        'center_scale': ('Center/Scale', 'tab:brown'),
    }

    for cond, (label, color) in cond_types.items():
        for name, hist in histories.items():
            if cond in name.lower() and 'mnist' in name.lower():
                if 'rank4' in name.lower() and 'mid' in name.lower():
                    losses = hist.get('loss', [])
                    if losses:
                        epochs = range(1, len(losses) + 1)
                        ax.plot(epochs, losses, label=label, color=color,
                               linewidth=2, marker='o', markersize=4)
                        break

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Conditioning Type Ablation (LoRA rank=4, mid-block)')
    ax.legend(loc='upper right')
    ax.set_xlim(left=1)

    save_path = os.path.join(output_dir, 'conditioning_ablation.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mask_percentage_ablation(histories, output_dir):
    """Plot inpainting mask percentage ablation"""
    fig, ax = plt.subplots(figsize=(10, 6))

    percentages = [20, 40, 50, 60, 80]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(percentages)))

    for i, pct in enumerate(percentages):
        for name, hist in histories.items():
            if 'inpainting' in name.lower() and f'_{pct}_' in name:
                losses = hist.get('loss', [])
                if losses:
                    epochs = range(1, len(losses) + 1)
                    ax.plot(epochs, losses, label=f'{pct}% masked',
                           color=colors[i], linewidth=2, marker='o', markersize=4)
                    break

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Inpainting: Mask Percentage Ablation')
    ax.legend(title='Mask Coverage', loc='upper right')
    ax.set_xlim(left=1)

    save_path = os.path.join(output_dir, 'mask_percentage_ablation.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_dataset_comparison(histories, output_dir):
    """Plot comparison across datasets (MNIST, SVHN, CLEVR)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = {
        'mnist': ('MNIST (28x28, Gray)', 'tab:blue'),
        'svhn': ('SVHN (32x32, RGB)', 'tab:orange'),
        'clevr': ('CLEVR (64x64, RGB)', 'tab:green'),
    }

    for dataset, (label, color) in datasets.items():
        for name, hist in histories.items():
            if dataset in name.lower() and 'edge' in name.lower():
                if 'lora' in name.lower() and 'rank4' in name.lower() and 'mid' in name.lower():
                    losses = hist.get('loss', [])
                    if losses:
                        epochs = range(1, len(losses) + 1)
                        ax.plot(epochs, losses, label=label, color=color,
                               linewidth=2, marker='o', markersize=4)
                        break

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Cross-Dataset Comparison (LoRA rank=4, mid-block, edge)')
    ax.legend(loc='upper right')
    ax.set_xlim(left=1)

    save_path = os.path.join(output_dir, 'dataset_comparison.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_loss_bars(histories, output_dir):
    """Plot bar chart of final losses for all experiments"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract final losses
    final_losses = {}
    for name, hist in histories.items():
        losses = hist.get('loss', [])
        if losses:
            final_losses[name] = losses[-1]

    if not final_losses:
        print("No data for final loss bar chart")
        return

    # Sort by loss value
    sorted_items = sorted(final_losses.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    losses = [item[1] for item in sorted_items]

    # Color by model type
    colors = []
    for name in names:
        if 'controlnet' in name.lower():
            colors.append('tab:blue')
        elif 'concat' in name.lower():
            colors.append('tab:orange')
        elif 'lora' in name.lower():
            colors.append('tab:green')
        else:
            colors.append('tab:gray')

    y_pos = np.arange(len(names))
    ax.barh(y_pos, losses, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Training Loss (MSE)')
    ax.set_title('Final Loss Comparison Across All Experiments')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='ControlNet'),
        Patch(facecolor='tab:orange', label='Concat'),
        Patch(facecolor='tab:green', label='LoRA'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'final_loss_comparison.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_parameter_efficiency(output_dir):
    """Plot parameter efficiency comparison"""
    import sys
    sys.path.insert(0, '.')

    try:
        from model import create_model, count_parameters

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Model configurations to compare
        configs = [
            ("ControlNet", {"model_type": "controlnet", "conditioning_channels": 1}),
            ("Input Concat", {"model_type": "concat", "conditioning_channels": 1}),
            ("LoRA r=1", {"model_type": "lora", "rank": 1, "conditioning_channels": 1}),
            ("LoRA r=2", {"model_type": "lora", "rank": 2, "conditioning_channels": 1}),
            ("LoRA r=4", {"model_type": "lora", "rank": 4, "conditioning_channels": 1}),
            ("LoRA r=8", {"model_type": "lora", "rank": 8, "conditioning_channels": 1}),
            ("LoRA r=16", {"model_type": "lora", "rank": 16, "conditioning_channels": 1}),
        ]

        names = []
        total_params = []
        trainable_params = []

        for name, kwargs in configs:
            try:
                model = create_model(**kwargs)
                total, trainable = count_parameters(model)
                names.append(name)
                total_params.append(total / 1e6)  # Convert to millions
                trainable_params.append(trainable / 1e6)
            except Exception as e:
                print(f"Error creating model {name}: {e}")

        x = np.arange(len(names))
        width = 0.35

        # Plot 1: Total vs Trainable parameters
        bars1 = ax1.bar(x - width/2, total_params, width, label='Total', color='tab:blue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, trainable_params, width, label='Trainable', color='tab:green', alpha=0.8)

        ax1.set_ylabel('Parameters (Millions)')
        ax1.set_title('Parameter Count Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(bottom=0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}M', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # Plot 2: Parameter efficiency (percentage trainable)
        efficiency = [t/tot * 100 for t, tot in zip(trainable_params, total_params)]
        colors = ['tab:blue' if 'ControlNet' in n or 'Concat' in n else 'tab:green' for n in names]
        bars3 = ax2.bar(x, efficiency, color=colors, alpha=0.8)

        ax2.set_ylabel('Trainable Parameters (%)')
        ax2.set_title('Parameter Efficiency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylim(0, 110)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')

        # Add value labels
        for bar, eff in zip(bars3, efficiency):
            ax2.annotate(f'{eff:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'parameter_efficiency.png')
        plt.savefig(save_path, facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    except ImportError:
        print("Could not import model module for parameter efficiency plot")


def create_summary_figure(histories, output_dir):
    """Create a comprehensive summary figure"""
    fig = plt.figure(figsize=(16, 12))

    # Create subplot grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Model comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models_to_plot = ['controlnet', 'concat', 'lora_mid', 'lora_all']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for model, color in zip(models_to_plot, colors):
        for name, hist in histories.items():
            if model.replace('_', '') in name.lower().replace('_', '') and 'edge' in name.lower() and 'mnist' in name.lower():
                losses = hist.get('loss', [])
                if losses:
                    epochs = range(1, len(losses) + 1)
                    ax1.plot(epochs, losses, label=model.replace('_', ' ').title(),
                            color=color, linewidth=2)
                    break

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('A) Model Comparison')
    ax1.legend(loc='upper right')

    # Plot 2: Rank ablation
    ax2 = fig.add_subplot(gs[0, 1])
    ranks = [1, 2, 4, 8, 16]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

    for i, rank in enumerate(ranks):
        for name, hist in histories.items():
            if f'rank{rank}' in name.lower() and 'mid' in name.lower() and 'mnist' in name.lower():
                losses = hist.get('loss', [])
                if losses:
                    epochs = range(1, len(losses) + 1)
                    ax2.plot(epochs, losses, label=f'r={rank}', color=colors[i], linewidth=2)
                    break

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('B) LoRA Rank Ablation')
    ax2.legend(loc='upper right', title='Rank')

    # Plot 3: Conditioning types
    ax3 = fig.add_subplot(gs[1, 0])
    cond_colors = {'edge': 'tab:blue', 'sobel': 'tab:orange', 'inpainting': 'tab:green'}

    for cond, color in cond_colors.items():
        for name, hist in histories.items():
            if cond in name.lower() and 'rank4' in name.lower() and 'mnist' in name.lower():
                losses = hist.get('loss', [])
                if losses:
                    epochs = range(1, len(losses) + 1)
                    ax3.plot(epochs, losses, label=cond.title(), color=color, linewidth=2)
                    break

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('C) Conditioning Type Comparison')
    ax3.legend(loc='upper right')

    # Plot 4: Dataset comparison
    ax4 = fig.add_subplot(gs[1, 1])
    datasets = {'mnist': 'tab:blue', 'svhn': 'tab:orange', 'clevr': 'tab:green'}

    for dataset, color in datasets.items():
        for name, hist in histories.items():
            if dataset in name.lower() and 'lora' in name.lower() and 'edge' in name.lower():
                losses = hist.get('loss', [])
                if losses:
                    epochs = range(1, len(losses) + 1)
                    ax4.plot(epochs, losses, label=dataset.upper(), color=color, linewidth=2)
                    break

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('D) Cross-Dataset Comparison')
    ax4.legend(loc='upper right')

    plt.suptitle('LoRA vs ControlNet for Diffusion Model Conditioning: Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    save_path = os.path.join(output_dir, 'summary_figure.png')
    plt.savefig(save_path, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("GENERATING PLOTS FROM EXPERIMENT RESULTS")
    print("="*60)

    # Load results
    histories, configs = load_results(args.results_dir)

    if not histories:
        print(f"No training histories found in {args.results_dir}/")
        print("Run experiments first with: ./run_experiments.sh")
        return

    print(f"Loaded {len(histories)} experiment histories")
    print(f"Loaded {len(configs)} experiment configs")

    # Generate all plots
    print("\nGenerating plots...")

    plot_model_comparison(histories, args.output_dir)
    plot_rank_ablation(histories, args.output_dir)
    plot_layer_ablation(histories, args.output_dir)
    plot_conditioning_ablation(histories, args.output_dir)
    plot_mask_percentage_ablation(histories, args.output_dir)
    plot_dataset_comparison(histories, args.output_dir)
    plot_final_loss_bars(histories, args.output_dir)
    plot_parameter_efficiency(args.output_dir)
    create_summary_figure(histories, args.output_dir)

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED!")
    print(f"Saved to: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
