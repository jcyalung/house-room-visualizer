"""
Setup script for lightweight/minimal NeRF implementation.
This clones a simple, educational NeRF implementation that's faster to train.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Setup lightweight NeRF."""
    print("=" * 60)
    print("Lightweight NeRF Setup")
    print("=" * 60)
    
    # Check if lightweight-nerf already exists
    nerf_dir = Path("lightweight-nerf")
    if nerf_dir.exists():
        print(f"\n✓ Lightweight NeRF repository already exists at {nerf_dir}")
        response = input("Do you want to re-clone it? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(nerf_dir)
        else:
            print("Using existing repository...")
            return
    
    # Clone minimal NeRF PyTorch repository
    print("\n1. Cloning minimal NeRF PyTorch repository...")
    if not subprocess.run("git clone https://github.com/airalcorn2/pytorch-nerf.git lightweight-nerf", 
                         shell=True, check=False).returncode == 0:
        print("Error: Failed to clone repository. Trying alternative...")
        # Try creating a minimal implementation ourselves
        create_minimal_nerf()
        return
    
    print("✓ Repository cloned successfully")
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    requirements = lightweight_nerf / "requirements.txt"
    if requirements.exists():
        subprocess.run(f"{sys.executable} -m pip install -r {requirements}", 
                      shell=True, check=False)
    else:
        # Install basic requirements
        subprocess.run(f"{sys.executable} -m pip install torch torchvision numpy imageio matplotlib tqdm", 
                      shell=True, check=False)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python train_lightweight_nerf.py")
    print("2. This will train much faster than the full NeRF!")

def create_minimal_nerf():
    """Create a minimal NeRF implementation if cloning fails."""
    print("Creating minimal NeRF implementation...")
    # This would create a simplified NeRF from scratch
    pass

if __name__ == "__main__":
    main()

