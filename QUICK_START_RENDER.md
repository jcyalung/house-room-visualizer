# Quick Start: Render Your Trained NeRF

You have a trained model (`lightweight_nerf_outputs/model.pth`). Here's how to use it:

## 🚀 Quick Commands

### 1. Render Novel Views (Recommended)
```bash
python render_nerf.py
```

This creates 30 novel view images in `nerf_renders/novel/`

### 2. Render with Video
```bash
python render_nerf.py --create_video
```

Creates both images and an MP4 video!

### 3. Render Test Set
```bash
python render_nerf.py --mode test
```

Renders all test images for comparison.

### 4. Visualize Results
```bash
python visualize_nerf.py
```

Creates comparison images and galleries.

## 📋 Full Example

```bash
# Step 1: Render novel views with video
python render_nerf.py --create_video --num_views 60

# Step 2: Visualize the results
python visualize_nerf.py --render_dir nerf_renders/novel --mode gallery

# Step 3: Compare with ground truth (if you rendered test set)
python visualize_nerf.py --render_dir nerf_renders/test --mode compare
```

## 🎯 What You Get

After running `render_nerf.py`:
- **Images**: `nerf_renders/novel/*.png` - Individual rendered views
- **Video**: `nerf_renders/novel_views.mp4` - Animated video (if `--create_video`)
- **Comparison**: `nerf_renders/comparison.png` - Side-by-side with ground truth
- **Gallery**: `nerf_renders/gallery.png` - Grid of rendered images

## ⚙️ Options

```bash
python render_nerf.py \
    --model_path lightweight_nerf_outputs/model.pth \
    --data_dir nerf_data \
    --output_dir nerf_renders \
    --mode novel \
    --num_views 30 \
    --create_video
```

- `--model_path`: Path to your `.pth` file
- `--data_dir`: Directory with transforms.json
- `--output_dir`: Where to save renders
- `--mode`: `novel`, `test`, or `both`
- `--num_views`: Number of novel views (default: 30)
- `--create_video`: Generate MP4 video

## 🎬 View Your Results

1. **Images**: Open `nerf_renders/novel/` folder
2. **Video**: Play `nerf_renders/novel_views.mp4`
3. **Gallery**: Open `nerf_renders/gallery.png`

That's it! Enjoy your NeRF renders! 🎨
