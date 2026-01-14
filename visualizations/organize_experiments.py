"""
Organize experiment visualizations into grouped folders.
"""
import os
import shutil
from collections import defaultdict

# Source directory
src_dir = r"C:\Users\mahmo\Desktop\control\experiment_results_complete\visualizations"

# Define experiment groups
groups = {
    "1_model_comparison_mnist": [
        "samples_baseline_rank4_mid_edge_mnist_best.png",
        "samples_concat_rank4_mid_edge_mnist_best.png",
        "samples_controlnet_rank4_mid_edge_mnist_best.png",
        "samples_lora_rank4_mid_edge_mnist_best.png",
    ],
    "2_rank_ablation_mnist": [
        "samples_lora_rank1_mid_edge_mnist_best.png",
        "samples_lora_rank2_mid_edge_mnist_best.png",
        "samples_lora_rank4_mid_edge_mnist_best.png",
        "samples_lora_rank8_mid_edge_mnist_best.png",
        "samples_lora_rank16_mid_edge_mnist_best.png",
    ],
    "3_layer_ablation_rank2": [
        "samples_lora_rank2_early_edge_mnist_best.png",
        "samples_lora_rank2_mid_edge_mnist_best.png",
        "samples_lora_rank2_all_edge_mnist_best.png",
    ],
    "4_layer_ablation_rank4": [
        "samples_lora_rank4_early_edge_mnist_best.png",
        "samples_lora_rank4_mid_edge_mnist_best.png",
        "samples_lora_rank4_all_edge_mnist_best.png",
    ],
    "5_layer_ablation_rank8": [
        "samples_lora_rank8_early_edge_mnist_best.png",
        "samples_lora_rank8_mid_edge_mnist_best.png",
        "samples_lora_rank8_all_edge_mnist_best.png",
    ],
    "6_conditioning_type_mnist": [
        "samples_lora_rank4_mid_edge_mnist_best.png",
        "samples_lora_rank4_mid_sobel_mnist_best.png",
        "samples_lora_rank4_mid_skeleton_mnist_best.png",
        "samples_lora_rank4_mid_stroke_thickness_mnist_best.png",
        "samples_lora_rank4_mid_center_scale_mnist_best.png",
    ],
    "7_mask_type_ablation": [
        "samples_lora_rank4_mid_inpainting_digit_percentage50_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_contiguous50_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_top50_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_bottom50_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_random_rect50_mnist_best.png",
    ],
    "8_mask_percentage_ablation": [
        "samples_lora_rank4_mid_inpainting_digit_percentage20_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_percentage40_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_percentage50_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_percentage60_mnist_best.png",
        "samples_lora_rank4_mid_inpainting_digit_percentage80_mnist_best.png",
    ],
    "9_inpainting_lora_vs_controlnet": [
        "samples_lora_rank4_mid_inpainting_digit_percentage50_mnist_best.png",
        "samples_controlnet_rank4_mid_inpainting_digit_percentage50_mnist_best.png",
    ],
    "10_svhn_experiments": [
        "samples_lora_rank4_mid_edge_svhn_best.png",
        "samples_controlnet_rank4_mid_edge_svhn_best.png",
        "samples_lora_rank4_mid_color_histogram_svhn_best.png",
        "samples_lora_rank4_mid_inpainting_random_rect50_svhn_best.png",
    ],
}

def organize():
    print("="*60)
    print("ORGANIZING EXPERIMENT VISUALIZATIONS")
    print("="*60)
    
    for group_name, files in groups.items():
        # Create group directory
        group_dir = os.path.join(src_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"\n📁 {group_name}/")
        
        for filename in files:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(group_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)  # copy (not move) to keep originals
                print(f"   ✓ {filename}")
            else:
                print(f"   ✗ {filename} (not found)")
    
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"\nCreated {len(groups)} experiment group folders.")
    print("Original files preserved in root directory.")

if __name__ == "__main__":
    organize()
