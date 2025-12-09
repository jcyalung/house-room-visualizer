"""
Project 3D pointcloud to 2D images from specific camera viewpoints.
Can compare with NeRF renders side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import imageio
from tqdm import tqdm


def read_ply(ply_path):
    """Read a PLY file and return points and colors."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors
    except ImportError:
        return read_ply_manual(ply_path)


def read_ply_manual(ply_path):
    """Manually read a binary PLY file."""
    import struct
    
    with open(ply_path, 'rb') as f:
        # Read header
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            if line == 'end_header':
                break
        
        # Read binary data
        point_format = '<dddBBB'
        point_size = struct.calcsize(point_format)
        
        points = np.zeros((n_vertices, 3), dtype=np.float64)
        colors = np.zeros((n_vertices, 3), dtype=np.float64)
        
        for i in range(n_vertices):
            data = f.read(point_size)
            if len(data) < point_size:
                break
            values = struct.unpack(point_format, data)
            points[i] = values[:3]
            colors[i] = np.array(values[3:6]) / 255.0
        
        return points, colors


def project_points_to_image(points, colors, c2w, K, H, W, near=0.1, far=10.0):
    """
    Project 3D points onto a 2D image plane.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-1)
        c2w: (4, 4) camera-to-world transformation matrix
        K: (3, 3) camera intrinsic matrix
        H, W: image height and width
        near, far: depth clipping planes
    
    Returns:
        2D image (H, W, 3) with projected points
    """
    # Convert c2w to w2c (world to camera)
    w2c = np.linalg.inv(c2w)
    
    # Transform points to camera coordinates
    points_hom = np.hstack([points, np.ones((len(points), 1))])
    points_cam = (w2c @ points_hom.T).T[:, :3]
    
    # Filter points behind the camera
    mask = points_cam[:, 2] > near
    mask &= points_cam[:, 2] < far
    points_cam = points_cam[mask]
    colors_valid = colors[mask] if colors is not None else None
    
    # Project to image plane
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    depths = points_cam[:, 2]
    
    # Filter points outside image bounds
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W)
    mask &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    
    points_2d = points_2d[mask]
    depths = depths[mask]
    colors_valid = colors_valid[mask] if colors_valid is not None else None
    
    # Create image with depth sorting (painter's algorithm)
    image = np.zeros((H, W, 3), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf)
    
    # Sort by depth (far to near)
    sort_idx = np.argsort(-depths)
    points_2d = points_2d[sort_idx]
    depths = depths[sort_idx]
    if colors_valid is not None:
        colors_valid = colors_valid[sort_idx]
    
    # Rasterize points (with small splat size for visibility)
    splat_size = 1  # Pixel radius for each point
    for i in range(len(points_2d)):
        x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
        d = depths[i]
        
        for dx in range(-splat_size, splat_size + 1):
            for dy in range(-splat_size, splat_size + 1):
                px, py = x + dx, y + dy
                if 0 <= px < W and 0 <= py < H:
                    if d < depth_buffer[py, px]:
                        depth_buffer[py, px] = d
                        if colors_valid is not None:
                            image[py, px] = colors_valid[i]
                        else:
                            image[py, px] = [0.7, 0.7, 0.7]
    
    return image, depth_buffer


def render_pointcloud_from_poses(ply_path, transforms_path, output_dir, 
                                  max_views=30, downsample=None):
    """
    Render 2D images from pointcloud using camera poses from transforms.json.
    """
    print(f"Loading pointcloud from {ply_path}...")
    points, colors = read_ply(ply_path)
    print(f"✓ Loaded {len(points):,} points")
    
    if downsample and downsample < len(points):
        print(f"Downsampling to {downsample:,} points...")
        idx = np.random.choice(len(points), downsample, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None
    
    # Load camera transforms
    print(f"Loading camera poses from {transforms_path}...")
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
    
    H, W = meta['h'], meta['w']
    camera_angle_x = meta['camera_angle_x']
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = meta['frames'][:max_views]
    print(f"Rendering {len(frames)} views...")
    
    for i, frame in enumerate(tqdm(frames, desc="Projecting")):
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        image, _ = project_points_to_image(points, colors, c2w, K, H, W)
        
        # Save image
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        output_path = output_dir / f"pointcloud_{i:04d}.png"
        imageio.imwrite(output_path, image_uint8)
    
    print(f"✓ Pointcloud renders saved to {output_dir}")


def render_orbit_views(ply_path, output_dir, num_views=30, H=738, W=994,
                       focal=500.0, radius=3.0, height=1.5, downsample=None):
    """
    Render 2D images from pointcloud using an orbital camera trajectory.
    """
    print(f"Loading pointcloud from {ply_path}...")
    points, colors = read_ply(ply_path)
    print(f"✓ Loaded {len(points):,} points")
    
    if downsample and downsample < len(points):
        print(f"Downsampling to {downsample:,} points...")
        idx = np.random.choice(len(points), downsample, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {num_views} orbit views...")
    
    for i in tqdm(range(num_views), desc="Projecting"):
        angle = 2 * np.pi * i / num_views
        
        # Camera position
        x = radius * np.cos(angle)
        y = height
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
        
        image, _ = project_points_to_image(points, colors, c2w, K, H, W)
        
        # Save image
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        output_path = output_dir / f"pointcloud_{i:04d}.png"
        imageio.imwrite(output_path, image_uint8)
    
    print(f"✓ Orbit views saved to {output_dir}")


def create_comparison(pointcloud_dir, nerf_dir, output_path, num_images=5):
    """
    Create side-by-side comparison of pointcloud projections and NeRF renders.
    """
    pointcloud_dir = Path(pointcloud_dir)
    nerf_dir = Path(nerf_dir)
    
    pc_images = sorted(pointcloud_dir.glob("*.png"))
    nerf_images = sorted(nerf_dir.glob("*.png"))
    
    num_images = min(num_images, len(pc_images), len(nerf_images))
    
    fig, axes = plt.subplots(num_images, 2, figsize=(14, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Pointcloud projection
        pc_img = imageio.imread(pc_images[i])
        axes[i, 0].imshow(pc_img)
        axes[i, 0].set_title(f"Pointcloud Projection {i}")
        axes[i, 0].axis('off')
        
        # NeRF render
        nerf_img = imageio.imread(nerf_images[i])
        axes[i, 1].imshow(nerf_img)
        axes[i, 1].set_title(f"NeRF Render {i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to {output_path}")
    plt.close()


def create_video(image_dir, output_path, fps=10):
    """Create a video from rendered images."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.png"))
    
    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Creating video from {len(images)} images...")
    frames = []
    for img_path in tqdm(images, desc="Loading"):
        img = imageio.imread(img_path)
        frames.append(img)
    
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"✓ Video saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Project pointcloud to 2D images")
    parser.add_argument("--ply_path", type=str, 
                       default="room_datasets/coffee_room/iphone/long_capture/polycam_pointcloud.ply",
                       help="Path to PLY file")
    parser.add_argument("--transforms_path", type=str,
                       default="nerf_data/transforms_train.json",
                       help="Path to transforms JSON for camera poses")
    parser.add_argument("--output_dir", type=str, default="pointcloud_renders",
                       help="Output directory for rendered images")
    parser.add_argument("--mode", type=str, default="orbit",
                       choices=["poses", "orbit", "compare"],
                       help="Mode: 'poses' uses transforms.json, 'orbit' creates circular path, 'compare' creates comparison")
    parser.add_argument("--num_views", type=int, default=30,
                       help="Number of views to render")
    parser.add_argument("--downsample", type=int, default=200000,
                       help="Downsample points for faster rendering")
    parser.add_argument("--nerf_dir", type=str, default="nerf_renders/novel",
                       help="Directory with NeRF renders for comparison")
    parser.add_argument("--create_video", action="store_true",
                       help="Create video from rendered images")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pointcloud 2D Projection")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    
    if args.mode == "poses":
        render_pointcloud_from_poses(
            args.ply_path, 
            args.transforms_path,
            output_dir / "from_poses",
            max_views=args.num_views,
            downsample=args.downsample
        )
        render_dir = output_dir / "from_poses"
        
    elif args.mode == "orbit":
        render_orbit_views(
            args.ply_path,
            output_dir / "orbit",
            num_views=args.num_views,
            downsample=args.downsample
        )
        render_dir = output_dir / "orbit"
        
    elif args.mode == "compare":
        # First render orbit views
        render_orbit_views(
            args.ply_path,
            output_dir / "orbit",
            num_views=args.num_views,
            downsample=args.downsample
        )
        # Then create comparison
        create_comparison(
            output_dir / "orbit",
            args.nerf_dir,
            output_dir / "comparison.png",
            num_images=5
        )
        render_dir = output_dir / "orbit"
    
    if args.create_video:
        create_video(render_dir, output_dir / "pointcloud_orbit.mp4", fps=10)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
