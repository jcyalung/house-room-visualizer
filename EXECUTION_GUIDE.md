# Program Execution Guide

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy opencv-python open3d
```

---

## Program Execution

### Option 1: Synthetic Reconstruction (Recommended for Testing)

This runs the complete pipeline with synthetic data (no real captures needed):

```bash
python synthetic_reconstruction.py
```

**What it does:**
1. Creates a synthetic 2×2×2 meter cardboard room
2. Projects points into virtual camera and projector
3. Triangulates to reconstruct 3D points
4. Visualizes the point cloud in an interactive Open3D window

**Expected output:**
- Prints example camera/projector pixels
- Prints reconstructed point cloud shape
- Opens interactive 3D visualization window
- Use mouse to rotate/zoom the point cloud

---

### Option 2: Alternative Synthetic Reconstruction

Similar to Option 1, but with different filtering:

```bash
python reconstruct_room.py
```

**What it does:**
- Same as `synthetic_reconstruction.py` but uses different point cloud cleaning
- Filters points to only those visible in both camera and projector FOVs

---

### Option 3: Generate Gray-Code Patterns

Generate the structured-light patterns for projection:

```bash
python graycode.py
```

**What it does:**
- Generates binary stripe patterns (default: 10 patterns for 1024×768 resolution)
- Prints the number of patterns generated
- Patterns are stored in memory (modify to save images if needed)

**To save patterns to files, modify `graycode.py`:**
```python
patterns = generate_gray_code_patterns()
for i, pattern in enumerate(patterns):
    cv2.imwrite(f'pattern_{i:02d}.png', pattern)
```

---

### Option 4: Real Data Reconstruction (Requires Implementation)

**⚠️ This is a template and needs implementation:**

```bash
python real_reconstruction_template.py
```

**Current status:** Will fail with `NotImplementedError`

**What needs to be done:**
1. Replace placeholder calibration data (lines 10-15) with real camera/projector intrinsics/extrinsics
2. Implement `decode_graycode_dataset()` function (line 24-25) to:
   - Load captured Gray-code images from `room_datasets/`
   - Decode each camera pixel to get corresponding projector pixel
   - Return `(cam_pixels, proj_pixels)` as N×2 numpy arrays

---

## Quick Start (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run synthetic reconstruction:**
   ```bash
   python synthetic_reconstruction.py
   ```

3. **Interact with the 3D visualization:**
   - **Rotate:** Left-click and drag
   - **Zoom:** Scroll wheel
   - **Pan:** Right-click and drag
   - **Close:** Press 'Q' or close the window

---

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure you're running from the project root directory

### Open3D Window Not Appearing
- On macOS/Linux: May need X11 forwarding if using SSH
- On Windows: Should work out of the box
- Try updating Open3D: `pip install --upgrade open3d`

### Memory Issues
- Reduce `resolution` in `synthetic_room.py` (line 14) from 120 to a smaller value
- Increase `voxel_size` in `synthetic_reconstruction.py` (line 68) from 0.05 to 0.1

---

## File Structure

```
house-room-visualizer/
├── requirements.txt              # Dependencies
├── graycode.py                   # Generate patterns
├── decoder.py                    # Decode patterns
├── project_points.py             # 3D→2D projection
├── triangulation.py              # 3D reconstruction
├── synthetic_room.py             # Synthetic geometry
├── syntheticcam.py               # Camera/projector models
├── synthetic_reconstruction.py   # ⭐ Main synthetic demo
├── reconstruct_room.py           # Alternative synthetic demo
├── real_reconstruction_template.py  # Template for real data
└── room_datasets/                # Real captured data (when available)
```
