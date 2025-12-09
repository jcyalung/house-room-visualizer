# Using Your Trained NeRF Model

You've trained a lightweight NeRF model and have a `.pth` file! Here's what you can do with it:

## 🎨 Render Novel Views

Generate new views of your scene from different camera angles:

```bash
python render_nerf.py --model_path lightweight_nerf_outputs/model.pth
```

This will:
- Render novel views in a circular trajectory (30 views by default)
- Save images to `nerf_renders/novel/`
- Optionally create a video: `--create_video`

**Options:**
- `--mode novel` - Render novel views (default)
- `--mode test` - Render test set images
- `--mode both` - Render both
- `--num_views 60` - Number of novel views (default: 30)
- `--create_video` - Create MP4 video from renders

## 📊 Visualize Results

Compare your renders with ground truth images:

```bash
python visualize_nerf.py --render_dir nerf_renders/test
```

This creates:
- `comparison.png` - Side-by-side comparison of renders vs ground truth
- `gallery.png` - Gallery of rendered images

## 🎬 Create Videos

Generate a video of novel views:

```bash
python render_nerf.py --model_path lightweight_nerf_outputs/model.pth --create_video
```

This creates `nerf_renders/novel_views.mp4` showing the scene from different angles.

## 📁 Output Structure

After rendering, you'll have:

```
nerf_renders/
├── novel/              # Novel view renders
│   ├── novel_0000.png
│   ├── novel_0001.png
│   └── ...
├── test/               # Test set renders
│   ├── test_0000.png
│   └── ...
├── novel_views.mp4     # Video (if created)
└── comparison.png      # Comparison visualization
```

## 🔧 Advanced Usage

### Render Specific Camera Poses

You can modify `render_nerf.py` to render from specific camera poses if you have them.

### Export Point Cloud

To extract a point cloud from the NeRF:

```python
# You can add this to render_nerf.py
# Sample points and extract density to create point cloud
```

### Adjust Rendering Quality

In `render_nerf.py`, you can adjust:
- `N_samples` - More samples = better quality but slower (default: 64)
- `chunk` - Memory usage (default: 1024*32)
- `near`/`far` - Depth range (default: 2.0, 6.0)

## 📝 Example Workflow

1. **Render novel views:**
   ```bash
   python render_nerf.py --model_path lightweight_nerf_outputs/model.pth --create_video
   ```

2. **Visualize results:**
   ```bash
   python visualize_nerf.py --render_dir nerf_renders/novel --mode gallery
   ```

3. **Compare with ground truth:**
   ```bash
   python visualize_nerf.py --render_dir nerf_renders/test --mode compare
   ```

## 🐛 Troubleshooting

**Renders are black/dark:**
- Model might not have trained long enough
- Try retraining with more iterations: `python train_lightweight_nerf.py --N_iters 20000`

**Renders are blurry:**
- Increase `N_samples` in `render_nerf.py` (e.g., 128 instead of 64)
- Model might need more training

**Out of memory:**
- Reduce `chunk` size in `render_nerf.py`
- Render smaller images or use CPU

## 🎯 Next Steps

1. **Improve quality**: Train longer or use the full NeRF implementation
2. **Export mesh**: Convert NeRF to mesh using marching cubes
3. **Interactive viewer**: Create a web-based viewer for your NeRF
4. **Fine-tune**: Adjust hyperparameters for better results

Enjoy your NeRF! 🚀
