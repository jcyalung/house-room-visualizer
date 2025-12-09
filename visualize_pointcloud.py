"""
3D Point Cloud Visualizer for PLY files.
Uses Open3D for interactive 3D visualization.
"""

import argparse
import numpy as np
from pathlib import Path

def load_and_visualize_open3d(ply_path, downsample=None, point_size=2.0):
    """Visualize pointcloud using Open3D (interactive 3D viewer)."""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "open3d"])
        import open3d as o3d
    
    print(f"Loading pointcloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    n_points = len(pcd.points)
    print(f"✓ Loaded {n_points:,} points")
    
    # Optionally downsample for performance
    if downsample and downsample < n_points:
        print(f"Downsampling to {downsample:,} points for better performance...")
        indices = np.random.choice(n_points, downsample, replace=False)
        pcd = pcd.select_by_index(indices)
        print(f"✓ Downsampled to {len(pcd.points):,} points")
    
    # Get bounding box info
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    print(f"Bounding box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"Bounding box extent: ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})")
    
    # Create visualizer
    print("\n" + "=" * 60)
    print("Opening 3D Viewer...")
    print("=" * 60)
    print("\nControls:")
    print("  - Left click + drag: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Middle click + drag: Pan")
    print("  - 'R': Reset view")
    print("  - 'Q' or ESC: Quit")
    print("  - '+'/'-': Increase/decrease point size")
    print("=" * 60 + "\n")
    
    # Set up visualization with better defaults
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Room Point Cloud Viewer", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Get render options and set point size
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Set a good initial viewpoint
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()
    
    print("✓ Visualization closed")


def load_and_visualize_matplotlib(ply_path, downsample=50000, figsize=(12, 10)):
    """Visualize pointcloud using matplotlib (static 3D plot)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    try:
        import open3d as o3d
        print(f"Loading pointcloud from {ply_path}...")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    except ImportError:
        # Fallback: read PLY manually
        print(f"Loading pointcloud from {ply_path} (manual parser)...")
        points, colors = read_ply_manual(ply_path)
    
    n_points = len(points)
    print(f"✓ Loaded {n_points:,} points")
    
    # Downsample for matplotlib (it can't handle millions of points)
    if downsample and downsample < n_points:
        print(f"Downsampling to {downsample:,} points for matplotlib...")
        indices = np.random.choice(n_points, downsample, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    print("Creating 3D scatter plot...")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=0.5, alpha=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   s=0.5, alpha=0.8, c='steelblue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Room Point Cloud ({len(points):,} points)')
    
    # Equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    # Save and show
    output_path = Path(ply_path).parent / "pointcloud_view.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Static view saved to {output_path}")
    
    plt.show()


def read_ply_manual(ply_path):
    """Manually read a PLY file (fallback if Open3D not available)."""
    import struct
    
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[1], parts[2]))
        
        print(f"Found {n_vertices:,} vertices with properties: {[p[1] for p in properties]}")
        
        # Read binary data
        # Format: x, y, z (double), r, g, b (uchar)
        point_format = '<dddBBB'  # Little endian: 3 doubles + 3 unsigned chars
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
            
            if (i + 1) % 100000 == 0:
                print(f"  Read {i+1:,} / {n_vertices:,} points...")
        
        return points, colors


def create_video_orbit(ply_path, output_path="pointcloud_orbit.mp4", n_frames=120, 
                       downsample=None, resolution=(1280, 720)):
    """Create an orbiting video around the pointcloud."""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D required for video creation. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "open3d"])
        import open3d as o3d
    
    print(f"Loading pointcloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    n_points = len(pcd.points)
    print(f"✓ Loaded {n_points:,} points")
    
    if downsample and downsample < n_points:
        print(f"Downsampling to {downsample:,} points...")
        indices = np.random.choice(n_points, downsample, replace=False)
        pcd = pcd.select_by_index(indices)
    
    # Create offscreen renderer
    print(f"Creating orbit video with {n_frames} frames...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=resolution[0], height=resolution[1])
    vis.add_geometry(pcd)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Get view control
    view_ctl = vis.get_view_control()
    
    # Render frames
    frames = []
    for i in range(n_frames):
        # Rotate camera
        view_ctl.rotate(10.0, 0.0)  # Horizontal rotation
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        img = vis.capture_screen_float_buffer(do_render=True)
        frames.append((np.asarray(img) * 255).astype(np.uint8))
        
        if (i + 1) % 30 == 0:
            print(f"  Rendered {i+1}/{n_frames} frames...")
    
    vis.destroy_window()
    
    # Save video
    print(f"Saving video to {output_path}...")
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=30)
        print(f"✓ Video saved to {output_path}")
    except Exception as e:
        print(f"Could not save video: {e}")
        # Save as GIF instead
        gif_path = output_path.replace('.mp4', '.gif')
        imageio.mimwrite(gif_path, frames[::2], fps=15)  # Skip frames for smaller GIF
        print(f"✓ GIF saved to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="3D Point Cloud Visualizer")
    parser.add_argument("--ply_path", type=str, 
                       default="room_datasets/coffee_room/iphone/long_capture/polycam_pointcloud.ply",
                       help="Path to PLY file")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "static", "video"],
                       help="Visualization mode")
    parser.add_argument("--downsample", type=int, default=None,
                       help="Downsample to N points (default: None for interactive, 50000 for static)")
    parser.add_argument("--point_size", type=float, default=2.0,
                       help="Point size for rendering")
    parser.add_argument("--output", type=str, default="pointcloud_orbit.mp4",
                       help="Output path for video mode")
    
    args = parser.parse_args()
    
    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        print(f"Error: PLY file not found at {ply_path}")
        return
    
    print("=" * 60)
    print("3D Point Cloud Visualizer")
    print("=" * 60)
    print(f"PLY file: {ply_path}")
    print(f"Mode: {args.mode}")
    print("=" * 60 + "\n")
    
    if args.mode == "interactive":
        load_and_visualize_open3d(ply_path, downsample=args.downsample, 
                                   point_size=args.point_size)
    elif args.mode == "static":
        downsample = args.downsample or 50000
        load_and_visualize_matplotlib(ply_path, downsample=downsample)
    elif args.mode == "video":
        create_video_orbit(ply_path, output_path=args.output, 
                          downsample=args.downsample)


if __name__ == "__main__":
    main()
