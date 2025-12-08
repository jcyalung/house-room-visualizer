# NeRF Integration Guide

This guide explains how to integrate Neural Radiance Fields (NeRFs) into your 3D reconstruction pipeline as an alternative or complement to structured-light triangulation.

## Overview

**NeRFs (Neural Radiance Fields)** learn a continuous 3D representation of a scene from multiple viewpoint images. Unlike triangulation-based methods that require structured light patterns, NeRFs can work with regular photographs from different angles.

### Key Differences

| Method | Input | Output | Pros | Cons |
|--------|------|--------|------|------|
| **Triangulation** | Gray-code patterns + projector | Point cloud | Fast, precise depth | Requires projector setup |
| **NeRF** | Multiple photos from different views | Continuous 3D representation | High-quality rendering, no special hardware | Slower training, requires many images |

## Installation

### Option 1: Using nerfstudio (Recommended)

```bash
# Install nerfstudio
pip install nerfstudio

# Verify installation
ns-train --help
```

### Option 2: Manual Installation

```bash
pip install torch torchvision
pip install nerfstudio
```

**Note:** For GPU acceleration (highly recommended), install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data Requirements

NeRFs require:
1. **Multiple images** from different viewpoints (typically 20-200 images)
2. **Camera poses** (extrinsics) for each image
3. **Camera intrinsics** (focal length, principal point)

Your existing `transformations.json` format is already compatible! It should contain:
- `frames`: Array of camera frames
- Each frame with:
  - `file_path`: Path to image
  - `transform_matrix`: 4x4 camera-to-world transformation matrix
  - `fl_x`, `fl_y`: Focal lengths
  - `cx`, `cy`: Principal point
  - `w`, `h`: Image dimensions

## Usage

### 1. Train a NeRF Model

```bash
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --max_iterations 30000
```

This will:
- Load images and camera poses from `transformations.json`
- Train a NeRF model (takes 30-60 minutes on GPU, hours on CPU)
- Save checkpoint to `outputs/` directory

### 2. Extract 3D Mesh

After training, extract a mesh:

```bash
python nerf_reconstruction.py \
    --extract_mesh \
    --checkpoint_path ./outputs/coffee_room/nerfacto/2024-01-01_123456/nerfstudio_models/step-000029999.ckpt \
    --mesh_output nerf_mesh.ply
```

### 3. Render Novel Views

Generate new viewpoints:

```bash
python nerf_reconstruction.py \
    --render_views \
    --checkpoint_path ./outputs/.../step-000029999.ckpt
```

### 4. Compare with Triangulation

Compare NeRF reconstruction with your triangulation-based method:

```bash
python nerf_reconstruction.py \
    --compare \
    --checkpoint_path ./outputs/.../step-000029999.ckpt \
    --triangulation_mesh coffee_room_mesh.ply \
    --mesh_output nerf_mesh.ply
```

## Integration with Existing Pipeline

### Option A: Replace Triangulation

Use NeRF instead of structured-light triangulation:

```python
# Instead of:
# points = triangulate(cam_pixels, proj_pixels, ...)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# Use:
# 1. Train NeRF (via command line or subprocess)
# 2. Extract mesh from NeRF
mesh = o3d.io.read_triangle_mesh("nerf_mesh.ply")
```

### Option B: Hybrid Approach

Combine both methods:
1. Use triangulation for initial sparse reconstruction
2. Use NeRF to fill in details and improve quality
3. Merge point clouds/meshes

### Option C: NeRF for Refinement

1. Generate initial mesh with triangulation
2. Use NeRF to refine surface details
3. Extract high-quality mesh from NeRF

## Advanced Usage

### Using Instant-NGP (Faster Training)

For faster training (requires CUDA GPU):

```bash
python nerf_reconstruction.py \
    --method instant-ngp \
    --data_path ./room_datasets/coffee_room/iphone/long_capture
```

### Custom Training Parameters

Modify `nerf_reconstruction.py` to adjust:
- Learning rate
- Batch size
- Number of training iterations
- Model architecture (nerfacto, instant-ngp, mipnerf, etc.)

### Interactive Viewer

During training, nerfstudio opens an interactive viewer. You can:
- Rotate around the scene
- Adjust rendering settings
- Export meshes directly from the viewer

## Troubleshooting

### Out of Memory

- Reduce `num_points` in mesh extraction
- Use smaller image resolution
- Train with fewer iterations

### Slow Training

- Use GPU acceleration (CUDA)
- Try `instant-ngp` model (faster)
- Reduce number of training images
- Lower `max_iterations`

### Poor Quality Results

- Ensure sufficient camera coverage (images from many angles)
- Check camera pose accuracy
- Increase training iterations
- Try different NeRF models (nerfacto, mipnerf, etc.)

### Import Errors

```bash
# Reinstall nerfstudio
pip install --upgrade nerfstudio

# Or install dependencies manually
pip install torch torchvision tqdm rich
```

## Performance Comparison

Expected performance on typical room dataset:

| Method | Training Time | Mesh Quality | Novel Views |
|--------|--------------|--------------|-------------|
| Triangulation | < 1 minute | Good | No |
| NeRF (CPU) | 2-4 hours | Excellent | Yes |
| NeRF (GPU) | 30-60 minutes | Excellent | Yes |

## Next Steps

1. **Train your first NeRF**: Use your existing `coffee_room` dataset
2. **Compare results**: See how NeRF compares to triangulation
3. **Optimize**: Adjust parameters for your specific use case
4. **Hybrid approach**: Combine both methods for best results

## References

- [nerfstudio Documentation](https://docs.nerf.studio/)
- [NeRF Paper](https://www.matthewtancik.com/nerf)
- [Instant-NGP Paper](https://nvlabs.github.io/instant-ngp/)

## Example Workflow

```bash
# 1. Validate data format
python nerf_reconstruction.py --data_path ./room_datasets/coffee_room/iphone/long_capture

# 2. Train NeRF (this takes time!)
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --max_iterations 30000

# 3. Extract mesh (after training completes)
python nerf_reconstruction.py \
    --extract_mesh \
    --checkpoint_path ./outputs/coffee_room/nerfacto/.../step-000029999.ckpt \
    --mesh_output coffee_room_nerf.ply

# 4. Visualize in Open3D
import open3d as o3d
mesh = o3d.io.read_triangle_mesh("coffee_room_nerf.ply")
o3d.visualization.draw_geometries([mesh])
```
