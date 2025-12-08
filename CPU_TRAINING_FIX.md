# Fixing CUDA Error on macOS/CPU

If you see this error:
```
AssertionError: Torch not compiled with CUDA enabled
```

This means nerfstudio is trying to use CUDA (GPU) but you're on macOS which doesn't support CUDA.

## Solution 1: Use Environment Variable (Recommended)

Before running the script, set this environment variable:

```bash
export CUDA_VISIBLE_DEVICES=''
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --max_iterations 5000
```

## Solution 2: Run with CPU Flag

The script now automatically adds `--machine.device-type cpu`, but if that doesn't work, try:

```bash
ns-train nerfacto \
    --data ./room_datasets/coffee_room/iphone/long_capture \
    --machine.device-type cpu \
    --max-num-iterations 5000
```

## Solution 3: Modify Config After Creation

If the above don't work:

1. Let nerfstudio create the config (it will fail)
2. Edit the config file: `outputs/long_capture/nerfacto/[timestamp]/config.yml`
3. Find `machine:` section and change:
   ```yaml
   machine:
     device_type: cpu  # Change from 'cuda' to 'cpu'
   ```
4. Resume training:
   ```bash
   ns-train nerfacto --load-config outputs/long_capture/nerfacto/[timestamp]/config.yml
   ```

## Important Notes

⚠️ **CPU training is VERY slow:**
- GPU: 30-60 minutes
- CPU: 5-20+ hours (depending on dataset size)

**Recommendations for CPU training:**
1. Reduce iterations: `--max_iterations 5000` (for testing)
2. Use fewer images (subsample your dataset)
3. Consider using a cloud GPU service (Google Colab, AWS, etc.)
4. Train overnight or when you don't need your computer

## Testing with Fewer Iterations

For quick testing on CPU:

```bash
python nerf_reconstruction.py \
    --method nerfstudio \
    --data_path ./room_datasets/coffee_room/iphone/long_capture \
    --max_iterations 1000
```

This will train much faster but quality will be lower. You can increase iterations later once you verify it works.
