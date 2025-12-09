"""
Training script for lightweight NeRF.
This is much faster than the full NeRF implementation.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lightweight_nerf import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train lightweight NeRF")
    parser.add_argument("--data_dir", type=str, default="nerf_data",
                       help="Directory containing NeRF data")
    parser.add_argument("--N_iters", type=int, default=10000,
                       help="Number of training iterations (default: 10000, much faster than full NeRF)")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="lightweight_nerf_outputs",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Lightweight NeRF Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Iterations: {args.N_iters}")
    print(f"Device: {device}")
    print("=" * 60)
    print("\nThis is a lightweight implementation that trains much faster!")
    print("Typical training time: 10-30 minutes (vs hours for full NeRF)")
    print("\n")
    
    # Load data
    print("Loading data...")
    try:
        images, poses, hwf = load_data(args.data_dir)
        print(f"Loaded {len(images)} images")
        print(f"Image size: {hwf[0]}x{hwf[1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create model
    print("\nCreating lightweight NeRF model...")
    model = LightweightNeRF(D=4, W=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    model = train(model, images, poses, hwf, args.N_iters, args.lr)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    print(f"\n✓ Model saved to {output_dir / 'model.pth'}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

