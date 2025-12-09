"""
Render novel views from a trained lightweight NeRF model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from lightweight_nerf import LightweightNeRF, get_rays, render_rays, load_data, raw2outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_image(model, H, W, K, c2w, near=2.0, far=6.0, N_samples=64, chunk=1024*32):
    """Render a single image from a camera pose."""
    model.eval()
    with torch.no_grad():
        # Get rays
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        
        # Flatten rays
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # Render in chunks
        rgb_map = []
        for i in tqdm(range(0, rays_o_flat.shape[0], chunk), desc="Rendering"):
            rays_o_chunk = rays_o_flat[i:i+chunk]
            rays_d_chunk = rays_d_flat[i:i+chunk]
            
            rgb, _, _ = render_rays(rays_o_chunk, rays_d_chunk, near, far, N_samples, model, chunk=chunk)
            rgb_map.append(rgb.cpu())
        
        rgb_map = torch.cat(rgb_map, 0)
        rgb_map = rgb_map.reshape(H, W, 3)
        
    return rgb_map.numpy()


def render_test_set(model, data_dir, output_dir):
    """Render all test set images."""
    print("Loading test data...")
    data_dir = Path(data_dir)
    
    with open(data_dir / "transforms_test.json", 'r') as f:
        meta = json.load(f)
    
    H, W = meta['h'], meta['w']
    camera_angle_x = meta['camera_angle_x']
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ], dtype=np.float32)
    K = torch.Tensor(K).to(device)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {len(meta['frames'])} test images...")
    for i, frame in enumerate(tqdm(meta['frames'], desc="Rendering test set")):
        c2w = torch.Tensor(frame['transform_matrix']).to(device)
        
        rgb = render_image(model, H, W, K, c2w)
        
        # Save image
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        output_path = output_dir / f"test_{i:04d}.png"
        imageio.imwrite(output_path, rgb_uint8)
    
    print(f"✓ Test images saved to {output_dir}")


def render_novel_views(model, data_dir, output_dir, num_views=30):
    """Render novel views in a circular trajectory."""
    print("Generating novel view trajectory...")
    data_dir = Path(data_dir)
    
    with open(data_dir / "transforms_train.json", 'r') as f:
        meta = json.load(f)
    
    H, W = meta['h'], meta['w']
    camera_angle_x = meta['camera_angle_x']
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ], dtype=np.float32)
    K = torch.Tensor(K).to(device)
    
    # Generate circular trajectory
    radius = 3.0
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position
        x = radius * np.cos(angle)
        y = 1.5  # Height
        z = radius * np.sin(angle)
        
        # Look at origin
        forward = np.array([0, 0, 0]) - np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create c2w matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = [x, y, z]
        c2w = torch.Tensor(c2w).to(device)
        
        # Render
        rgb = render_image(model, H, W, K, c2w)
        
        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        output_path = output_dir / f"novel_{i:04d}.png"
        imageio.imwrite(output_path, rgb_uint8)
    
    print(f"✓ Novel views saved to {output_dir}")


def create_video(image_dir, output_path, fps=10):
    """Create a video from rendered images."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.png"))
    
    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Creating video from {len(images)} images...")
    frames = []
    for img_path in tqdm(images, desc="Loading images"):
        img = imageio.imread(img_path)
        frames.append(img)
    
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"✓ Video saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Render from trained NeRF model")
    parser.add_argument("--model_path", type=str, default="lightweight_nerf_outputs/model.pth",
                       help="Path to trained model (.pth file)")
    parser.add_argument("--data_dir", type=str, default="nerf_data",
                       help="Directory containing NeRF data")
    parser.add_argument("--output_dir", type=str, default="nerf_renders",
                       help="Output directory for renders")
    parser.add_argument("--mode", type=str, default="novel", 
                       choices=["test", "novel", "both"],
                       help="What to render: test set, novel views, or both")
    parser.add_argument("--num_views", type=int, default=30,
                       help="Number of novel views to render")
    parser.add_argument("--create_video", action="store_true",
                       help="Create video from rendered images")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NeRF Rendering")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = LightweightNeRF(D=4, W=128).to(device)
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data for camera parameters
    print(f"\nLoading data from {args.data_dir}...")
    try:
        images, poses, hwf = load_data(args.data_dir)
        print(f"✓ Data loaded: {len(images)} images, size {hwf[0]}x{hwf[1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Render
    output_dir = Path(args.output_dir)
    
    if args.mode in ["test", "both"]:
        print("\n" + "=" * 60)
        print("Rendering test set...")
        print("=" * 60)
        render_test_set(model, args.data_dir, output_dir / "test")
    
    if args.mode in ["novel", "both"]:
        print("\n" + "=" * 60)
        print("Rendering novel views...")
        print("=" * 60)
        render_novel_views(model, args.data_dir, output_dir / "novel", args.num_views)
    
    # Create video if requested
    if args.create_video:
        print("\n" + "=" * 60)
        print("Creating video...")
        print("=" * 60)
        if args.mode in ["novel", "both"]:
            create_video(output_dir / "novel", output_dir / "novel_views.mp4", fps=10)
        if args.mode in ["test", "both"]:
            create_video(output_dir / "test", output_dir / "test_set.mp4", fps=5)
    
    print("\n" + "=" * 60)
    print("Rendering complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nYou can:")
    print("1. View individual images in the output directory")
    print("2. Open the video files (if created)")
    print("3. Compare renders with original images")

if __name__ == "__main__":
    main()
