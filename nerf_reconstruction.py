"""
NeRF-based 3D Reconstruction Integration

This script integrates Neural Radiance Fields (NeRFs) for 3D reconstruction
as an alternative/complement to the structured-light triangulation approach.

NeRFs learn a continuous 3D representation from multiple viewpoint images,
allowing for high-quality novel view synthesis and 3D reconstruction.

Requirements:
- Multiple images from different viewpoints
- Camera poses (extrinsics) for each image
- Camera intrinsics (focal length, principal point)

Usage:
    # Option 1: Train NeRF using nerfstudio (recommended)
    python nerf_reconstruction.py --method nerfstudio --data_path ./room_datasets/coffee_room/iphone/long_capture
    
    # Option 2: Train NeRF using instant-ngp
    python nerf_reconstruction.py --method instant-ngp --data_path ./room_datasets/coffee_room/iphone/long_capture
    
    # Option 3: Extract mesh from trained NeRF
    python nerf_reconstruction.py --extract_mesh --checkpoint_path ./outputs/coffee_room/nerfacto/2024-01-01_123456/nerfstudio_models/step-000029999.ckpt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple


def load_transforms_json(json_path: str) -> Dict:
    """Load camera transforms from nerfstudio/COLMAP format JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def validate_data_format(data_path: str) -> bool:
    """Validate that the data directory has the required structure."""
    transforms_path = os.path.join(data_path, 'transformations.json')
    
    if not os.path.exists(transforms_path):
        print(f"❌ Error: transformations.json not found at {transforms_path}")
        return False
    
    try:
        transforms = load_transforms_json(transforms_path)
        if 'frames' not in transforms:
            print("❌ Error: 'frames' key not found in transformations.json")
            return False
        
        num_frames = len(transforms['frames'])
        print(f"✅ Found {num_frames} camera frames")
        
        # Check if image files exist
        images_dir = os.path.join(data_path, 'images')
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"✅ Found {len(image_files)} image files")
        else:
            print(f"⚠️  Warning: images directory not found at {images_dir}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading transforms: {e}")
        return False


def prepare_nerfstudio_data(data_path: str) -> bool:
    """
    Prepare data directory for nerfstudio by creating transforms.json if needed.
    
    Nerfstudio expects 'transforms.json' but your data might have 'transformations.json'.
    This function creates a symlink or copy if needed.
    
    Returns:
        True if preparation successful, False otherwise
    """
    transforms_json = os.path.join(data_path, 'transforms.json')
    transformations_json = os.path.join(data_path, 'transformations.json')
    
    # If transforms.json already exists, we're good
    if os.path.exists(transforms_json):
        print(f"✅ Found transforms.json")
        return True
    
    # If transformations.json exists, create transforms.json
    if os.path.exists(transformations_json):
        print(f"📝 Creating transforms.json from transformations.json...")
        try:
            # Copy the file (safer than symlink for cross-platform compatibility)
            shutil.copy2(transformations_json, transforms_json)
            print(f"✅ Created transforms.json")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not create transforms.json: {e}")
            return False
    
    print(f"❌ Error: Neither transforms.json nor transformations.json found in {data_path}")
    return False


def train_nerf_nerfstudio(data_path: str, output_dir: str = None, 
                         model_name: str = "nerfacto", 
                         max_num_iterations: int = 30000) -> str:
    """
    Train NeRF using nerfstudio.
    
    Args:
        data_path: Path to directory containing transformations.json and images/
        output_dir: Output directory for training results
        model_name: NeRF model type (nerfacto, instant-ngp, mipnerf, etc.)
        max_num_iterations: Maximum training iterations
    
    Returns:
        Path to trained checkpoint
    """
    print(f"\n🚀 Training NeRF using nerfstudio ({model_name} model)...")
    print(f"📁 Data path: {data_path}")
    
    # Check if nerfstudio is installed
    try:
        import nerfstudio
    except ImportError:
        print("\n❌ Error: nerfstudio is not installed!")
        print("Install it with: pip install nerfstudio")
        print("Or follow: https://docs.nerf.studio/quickstart/installation.html")
        sys.exit(1)
    
    # Prepare data format for nerfstudio (create transforms.json if needed)
    if not prepare_nerfstudio_data(data_path):
        print("\n❌ Error: Failed to prepare data for nerfstudio")
        sys.exit(1)
    
    # Detect device (CUDA not available on macOS)
    import platform
    try:
        import torch
        device_available = torch.cuda.is_available()
    except:
        device_available = False
    
    if not device_available:
        print("⚠️  CUDA not available. Forcing CPU mode (training will be slower)")
        print("   On macOS, training may take several hours. Consider reducing iterations for testing.")
    
    # Prepare command
    cmd = [
        "ns-train",
        model_name,
        "--data", data_path,
        "--max-num-iterations", str(max_num_iterations),
        "--viewer.quit-on-train-completion", "True"
    ]
    
    # Force CPU mode if CUDA not available
    if not device_available:
        cmd.extend(["--machine.device-type", "cpu"])
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    print(f"\n📝 Running command: {' '.join(cmd)}\n")
    
    try:
        # Set environment to prevent CUDA usage if needed
        env = None
        if not device_available:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            # Prevent PyTorch from trying to initialize CUDA
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        subprocess.run(cmd, check=True, env=env)
        print("\n✅ NeRF training completed!")
        
        # Find the checkpoint (nerfstudio saves to outputs/)
        if output_dir:
            checkpoint_dir = os.path.join(output_dir, data_path.split('/')[-1], model_name)
        else:
            checkpoint_dir = os.path.join("outputs", data_path.split('/')[-1], model_name)
        
        # Find latest checkpoint
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir) 
                      if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if subdirs:
                latest = sorted(subdirs)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest, 
                                              "nerfstudio_models", "step-000029999.ckpt")
                if os.path.exists(checkpoint_path):
                    return checkpoint_path
        
        print("⚠️  Checkpoint path not automatically found. Check outputs/ directory.")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        if not device_available:
            print("\n💡 Troubleshooting tips for CPU mode:")
            print("   1. The --machine.device-type cpu flag may not be supported in your nerfstudio version")
            print("   2. Try setting environment variable: export CUDA_VISIBLE_DEVICES=''")
            print("   3. CPU training is very slow (can take 10+ hours). Consider:")
            print("      - Reducing iterations: --max_iterations 5000")
            print("      - Using fewer images")
            print("      - Training on a machine with GPU")
        return None
    except FileNotFoundError:
        print("\n❌ Error: 'ns-train' command not found!")
        print("Make sure nerfstudio is installed and in your PATH")
        sys.exit(1)


def train_nerf_instant_ngp(data_path: str, output_path: str = None) -> str:
    """
    Train NeRF using instant-ngp (via nerfstudio wrapper).
    
    This is faster than standard NeRF but requires CUDA.
    """
    return train_nerf_nerfstudio(data_path, output_path, model_name="instant-ngp")


def extract_mesh_from_nerf(checkpoint_path: str, output_mesh_path: str = "nerf_mesh.ply",
                          num_points: int = 1000000, remove_outliers: bool = True) -> str:
    """
    Extract a 3D mesh from a trained NeRF model.
    
    Args:
        checkpoint_path: Path to trained NeRF checkpoint
        output_mesh_path: Output path for PLY mesh file
        num_points: Number of points to sample for mesh extraction
        remove_outliers: Whether to remove statistical outliers
    
    Returns:
        Path to extracted mesh file
    """
    print(f"\n🔧 Extracting mesh from NeRF checkpoint: {checkpoint_path}")
    
    try:
        import nerfstudio
    except ImportError:
        print("❌ Error: nerfstudio is not installed!")
        sys.exit(1)
    
    # Use nerfstudio's mesh extraction
    cmd = [
        "ns-export", "poisson",
        "--load-config", checkpoint_path.replace("nerfstudio_models/step-000029999.ckpt", 
                                                 "config.yml"),
        "--output-dir", os.path.dirname(output_mesh_path),
        "--num-points", str(num_points),
        "--remove-outliers", str(remove_outliers).lower()
    ]
    
    print(f"📝 Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Mesh extracted to {output_mesh_path}")
        return output_mesh_path
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Mesh extraction failed: {e}")
        print("\nAlternative: Use nerfstudio viewer to export mesh manually")
        return None


def render_novel_views(checkpoint_path: str, output_dir: str = "rendered_views",
                      num_views: int = 8) -> None:
    """
    Render novel views from trained NeRF.
    
    Args:
        checkpoint_path: Path to trained NeRF checkpoint
        output_dir: Directory to save rendered images
        num_views: Number of novel views to render
    """
    print(f"\n🎨 Rendering {num_views} novel views...")
    
    try:
        import nerfstudio
    except ImportError:
        print("❌ Error: nerfstudio is not installed!")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "ns-render", "camera-path",
        "--load-config", checkpoint_path.replace("nerfstudio_models/step-000029999.ckpt",
                                                 "config.yml"),
        "--output-path", output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Rendered views saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Rendering failed: {e}")


def compare_with_triangulation(nerf_mesh_path: str, triangulation_mesh_path: str) -> None:
    """
    Compare NeRF reconstruction with triangulation-based reconstruction.
    
    Args:
        nerf_mesh_path: Path to NeRF-extracted mesh
        triangulation_mesh_path: Path to triangulation-based mesh
    """
    print("\n📊 Comparing NeRF vs Triangulation reconstructions...")
    
    nerf_mesh = o3d.io.read_triangle_mesh(nerf_mesh_path)
    tri_mesh = o3d.io.read_triangle_mesh(triangulation_mesh_path)
    
    print(f"\nNeRF Mesh:")
    print(f"  Vertices: {len(nerf_mesh.vertices)}")
    print(f"  Triangles: {len(nerf_mesh.triangles)}")
    
    print(f"\nTriangulation Mesh:")
    print(f"  Vertices: {len(tri_mesh.vertices)}")
    print(f"  Triangles: {len(tri_mesh.triangles)}")
    
    # Visualize both
    nerf_mesh.paint_uniform_color([1, 0, 0])  # Red
    tri_mesh.paint_uniform_color([0, 0, 1])  # Blue
    
    print("\n🖼️  Opening visualization (Red=NeRF, Blue=Triangulation)...")
    o3d.visualization.draw_geometries([nerf_mesh, tri_mesh])


def main():
    parser = argparse.ArgumentParser(
        description="NeRF-based 3D Reconstruction Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train NeRF with nerfstudio
  python nerf_reconstruction.py --method nerfstudio --data_path ./room_datasets/coffee_room/iphone/long_capture
  
  # Train with instant-ngp (faster, requires CUDA)
  python nerf_reconstruction.py --method instant-ngp --data_path ./room_datasets/coffee_room/iphone/long_capture
  
  # Extract mesh from trained NeRF
  python nerf_reconstruction.py --extract_mesh --checkpoint_path ./outputs/coffee_room/nerfacto/.../step-000029999.ckpt
  
  # Render novel views
  python nerf_reconstruction.py --render_views --checkpoint_path ./outputs/.../step-000029999.ckpt
        """
    )
    
    parser.add_argument("--method", type=str, choices=["nerfstudio", "instant-ngp"],
                       default="nerfstudio",
                       help="NeRF training method (default: nerfstudio)")
    parser.add_argument("--data_path", type=str,
                       help="Path to directory containing transformations.json and images/")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for training results")
    parser.add_argument("--max_iterations", type=int, default=30000,
                       help="Maximum training iterations (default: 30000)")
    parser.add_argument("--extract_mesh", action="store_true",
                       help="Extract mesh from trained NeRF")
    parser.add_argument("--checkpoint_path", type=str,
                       help="Path to trained NeRF checkpoint")
    parser.add_argument("--mesh_output", type=str, default="nerf_mesh.ply",
                       help="Output path for extracted mesh")
    parser.add_argument("--render_views", action="store_true",
                       help="Render novel views from trained NeRF")
    parser.add_argument("--compare", action="store_true",
                       help="Compare NeRF mesh with triangulation mesh")
    parser.add_argument("--triangulation_mesh", type=str,
                       help="Path to triangulation-based mesh for comparison")
    
    args = parser.parse_args()
    
    # Validate data format if training
    if args.data_path and not args.extract_mesh and not args.render_views:
        if not validate_data_format(args.data_path):
            sys.exit(1)
    
    # Train NeRF
    if args.data_path and not args.extract_mesh and not args.render_views:
        if args.method == "nerfstudio":
            checkpoint = train_nerf_nerfstudio(
                args.data_path, 
                args.output_dir,
                model_name="nerfacto",
                max_num_iterations=args.max_iterations
            )
        elif args.method == "instant-ngp":
            checkpoint = train_nerf_instant_ngp(args.data_path, args.output_dir)
        
        if checkpoint:
            print(f"\n✅ Training complete! Checkpoint: {checkpoint}")
            print(f"\nNext steps:")
            print(f"  1. Extract mesh: python nerf_reconstruction.py --extract_mesh --checkpoint_path {checkpoint}")
            print(f"  2. Render views: python nerf_reconstruction.py --render_views --checkpoint_path {checkpoint}")
    
    # Extract mesh
    if args.extract_mesh:
        if not args.checkpoint_path:
            print("❌ Error: --checkpoint_path required for mesh extraction")
            sys.exit(1)
        extract_mesh_from_nerf(args.checkpoint_path, args.mesh_output)
    
    # Render views
    if args.render_views:
        if not args.checkpoint_path:
            print("❌ Error: --checkpoint_path required for rendering")
            sys.exit(1)
        render_novel_views(args.checkpoint_path)
    
    # Compare reconstructions
    if args.compare:
        if not args.checkpoint_path or not args.triangulation_mesh:
            print("❌ Error: --checkpoint_path and --triangulation_mesh required for comparison")
            sys.exit(1)
        compare_with_triangulation(args.mesh_output, args.triangulation_mesh)


if __name__ == "__main__":
    main()
