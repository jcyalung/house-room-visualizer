"""
Visualize NeRF results - compare rendered images with ground truth.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio
import json

def compare_renders(data_dir, render_dir, num_comparisons=5):
    """Compare rendered images with ground truth."""
    data_dir = Path(data_dir)
    render_dir = Path(render_dir)
    
    # Load test data
    with open(data_dir / "transforms_test.json", 'r') as f:
        meta = json.load(f)
    
    render_images = sorted(render_dir.glob("*.png"))
    
    if len(render_images) == 0:
        print(f"No rendered images found in {render_dir}")
        return
    
    num_comparisons = min(num_comparisons, len(render_images), len(meta['frames']))
    
    fig, axes = plt.subplots(num_comparisons, 2, figsize=(12, 6*num_comparisons))
    if num_comparisons == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_comparisons):
        # Load ground truth
        frame = meta['frames'][i]
        gt_path = data_dir / frame['file_path'].replace('./', '')
        if gt_path.exists():
            gt_img = imageio.imread(gt_path)
            axes[i, 0].imshow(gt_img)
            axes[i, 0].set_title(f"Ground Truth {i}")
            axes[i, 0].axis('off')
        
        # Load render
        render_img = imageio.imread(render_images[i])
        axes[i, 1].imshow(render_img)
        axes[i, 1].set_title(f"NeRF Render {i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = render_dir.parent / "comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to {output_path}")
    plt.close()


def create_gallery(render_dir, output_path, grid_size=(5, 6)):
    """Create a gallery of rendered images."""
    render_dir = Path(render_dir)
    images = sorted(render_dir.glob("*.png"))
    
    if len(images) == 0:
        print(f"No images found in {render_dir}")
        return
    
    n_images = min(len(images), grid_size[0] * grid_size[1])
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(n_images):
        img = imageio.imread(images[i])
        axes[i].imshow(img)
        axes[i].set_title(f"View {i}", fontsize=8)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gallery saved to {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize NeRF results")
    parser.add_argument("--data_dir", type=str, default="nerf_data",
                       help="Directory containing ground truth data")
    parser.add_argument("--render_dir", type=str, default="nerf_renders/test",
                       help="Directory containing rendered images")
    parser.add_argument("--mode", type=str, default="compare",
                       choices=["compare", "gallery", "both"],
                       help="Visualization mode")
    parser.add_argument("--num_comparisons", type=int, default=5,
                       help="Number of comparisons to show")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NeRF Visualization")
    print("=" * 60)
    
    if args.mode in ["compare", "both"]:
        print("\nCreating comparison...")
        compare_renders(args.data_dir, args.render_dir, args.num_comparisons)
    
    if args.mode in ["gallery", "both"]:
        print("\nCreating gallery...")
        output_path = Path(args.render_dir).parent / "gallery.png"
        create_gallery(args.render_dir, output_path)
    
    print("\n✓ Visualization complete!")

if __name__ == "__main__":
    main()
