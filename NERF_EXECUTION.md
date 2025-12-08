# How to Execute NeRF with nerfstudio

## Prerequisites

### 1. Install nerfstudio

```bash
# Basic installation
pip install nerfstudio

# OR for GPU support (recommended - much faster):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nerfstudio
```

**Verify installation:**
```bash
ns-train --help
```

If this command works, nerfstudio is installed correctly.

---

## Step-by-Step Execution

### Step 1: Prepare Your Data

Your data should be in this structure:
```
room_datasets/coffee_room/iphone/long_capture/
├── transformations.json
└── images/
    ├── frame_00001.jpg
    ├── frame_00002.jpg
    └── ...
```

**Optional: Use the helper script to prepare data:**
```bash
python nerf_simple.py \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --action prepare \
    --output_dir ./nerf_data
```

This creates a clean `nerf_data/` directory ready for training.

---

### Step 2: Train the NeRF Model

**Basic command:**
```bash
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture
```

**With custom output directory:**
```bash
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --output_dir ./my_nerf_outputs \
    --max_iterations 30000
```

**What happens:**
1. ✅ Validates your data format
2. 🚀 Starts NeRF training (takes 30-60 minutes on GPU, hours on CPU)
3. 📊 Opens interactive viewer (you can watch training progress)
4. 💾 Saves checkpoint to `outputs/` directory

**During training:**
- An interactive viewer window will open
- You can rotate/zoom the scene in real-time
- Training progress is shown in the terminal
- Press 'Q' in the viewer to quit (training continues)

---

### Step 3: Extract 3D Mesh (After Training Completes)

Once training finishes, extract a mesh:

```bash
python nerf_reconstruction.py \
    --extract_mesh \
    --checkpoint_path ./outputs/coffee_room/nerfacto/[timestamp]/nerfstudio_models/step-000029999.ckpt \
    --mesh_output nerf_mesh.ply
```

**Finding the checkpoint path:**
After training, checkpoints are saved in:
```
outputs/
└── [dataset_name]/
    └── nerfacto/
        └── [timestamp]/
            └── nerfstudio_models/
                └── step-000029999.ckpt
```

**Tip:** Use tab completion or list the directory:
```bash
ls -R outputs/
```

---

### Step 4: Visualize the Mesh

```python
import open3d as o3d
mesh = o3d.io.read_triangle_mesh("nerf_mesh.ply")
o3d.visualization.draw_geometries([mesh])
```

Or use any 3D viewer (MeshLab, Blender, etc.)

---

## Alternative: Direct nerfstudio Commands

You can also use nerfstudio directly without the wrapper script:

### 1. Train NeRF
```bash
ns-train nerfacto \
    --data ./room_datasets/coffee_room/iphone/long_capture \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True
```

### 2. Extract Mesh
```bash
ns-export poisson \
    --load-config ./outputs/coffee_room/nerfacto/[timestamp]/config.yml \
    --output-dir ./meshes \
    --num-points 1000000
```

### 3. Render Novel Views
```bash
ns-render camera-path \
    --load-config ./outputs/coffee_room/nerfacto/[timestamp]/config.yml \
    --output-path ./rendered_views
```

---

## Quick Start (Copy-Paste Ready)

```bash
# 1. Install nerfstudio
pip install nerfstudio

# 2. Train NeRF (this takes time!)
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --max_iterations 30000

# 3. After training completes, find your checkpoint:
#    Check: outputs/coffee_room/nerfacto/[timestamp]/nerfstudio_models/

# 4. Extract mesh (replace [timestamp] with actual folder name):
python nerf_reconstruction.py \
    --extract_mesh \
    --checkpoint_path ./outputs/coffee_room/nerfacto/[timestamp]/nerfstudio_models/step-000029999.ckpt \
    --mesh_output coffee_room_nerf.ply

# 5. Visualize:
python -c "import open3d as o3d; mesh = o3d.io.read_triangle_mesh('coffee_room_nerf.ply'); o3d.visualization.draw_geometries([mesh])"
```

---

## Troubleshooting

### "ns-train: command not found"
```bash
# Reinstall nerfstudio
pip install --upgrade nerfstudio

# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### "transformations.json not found"
- Make sure you're pointing to the correct directory
- Check that `transformations.json` exists in that directory
- Use absolute path: `--data_path /full/path/to/directory`

### "CUDA out of memory"
- Reduce image resolution
- Use fewer training images
- Try `instant-ngp` model instead (more memory efficient)

### Training is too slow
- Use GPU acceleration (CUDA)
- Try `instant-ngp` model: `--method instant-ngp`
- Reduce `--max_iterations` (e.g., 10000 for testing)

### Can't find checkpoint after training
```bash
# List all outputs
find outputs/ -name "*.ckpt" -type f

# Or check the timestamp folder
ls -la outputs/coffee_room/nerfacto/
```

---

## Expected Output

**During training:**
```
✅ Found 100 camera frames
✅ Found 100 image files
🚀 Training NeRF using nerfstudio (nerfacto model)...
📁 Data path: ./room_datasets/coffee_room/iphone/long_capture

[Training progress logs...]
Step 1000/30000: Loss: 0.1234
Step 2000/30000: Loss: 0.0987
...
✅ NeRF training completed!
```

**After mesh extraction:**
```
🔧 Extracting mesh from NeRF checkpoint...
✅ Mesh extracted to nerf_mesh.ply
```

---

## Performance Tips

1. **Use GPU**: Training on GPU is 10-50x faster than CPU
2. **Start small**: Test with `--max_iterations 5000` first
3. **Monitor memory**: If OOM, reduce image count or resolution
4. **Use instant-ngp**: Faster training, similar quality: `--method instant-ngp`

---

## Next Steps

After extracting your mesh:
1. Compare with triangulation: Use `--compare` flag
2. Render novel views: Use `--render_views` flag  
3. Integrate into your pipeline: Load `nerf_mesh.ply` in your existing code
