"""
Simplified NeRF Integration - Alternative Approach

This script provides a simpler way to integrate NeRF concepts without requiring
the full nerfstudio installation. It demonstrates how to prepare data for NeRF
training and provides a bridge to external NeRF libraries.

For full NeRF training, use nerf_reconstruction.py with nerfstudio.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import open3d as o3d


def load_camera_data(json_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """
    Load camera poses and intrinsics from transformations.json.
    
    Returns:
        poses: List of 4x4 camera-to-world transformation matrices
        intrinsics: List of camera intrinsic parameters (fx, fy, cx, cy)
        metadata: Dictionary with image paths and other metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses = []
    intrinsics = []
    image_paths = []
    
    for frame in data['frames']:
        # Extract transformation matrix
        transform = np.array(frame['transform_matrix'])
        poses.append(transform)
        
        # Extract intrinsics
        K = np.array([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ])
        intrinsics.append({
            'K': K,
            'fx': frame['fl_x'],
            'fy': frame['fl_y'],
            'cx': frame['cx'],
            'cy': frame['cy'],
            'width': frame['w'],
            'height': frame['h']
        })
        
        # Extract image path
        image_paths.append(frame['file_path'])
    
    metadata = {
        'image_paths': image_paths,
        'camera_model': data.get('camera_model', 'OPENCV'),
        'num_frames': len(poses)
    }
    
    return poses, intrinsics, metadata


def convert_to_colmap_format(data_path: str, output_dir: str = "colmap_format"):
    """
    Convert transformations.json to COLMAP format (cameras.txt, images.txt, points3D.txt).
    
    This format is compatible with many NeRF implementations.
    """
    json_path = os.path.join(data_path, 'transformations.json')
    poses, intrinsics, metadata = load_camera_data(json_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write cameras.txt
    cameras_file = os.path.join(output_dir, 'cameras.txt')
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        # Use first camera's intrinsics (assuming all cameras are the same)
        K = intrinsics[0]
        f.write(f"1 PINHOLE {K['width']} {K['height']} {K['fx']} {K['fy']} {K['cx']} {K['cy']}\n")
    
    # Write images.txt
    images_file = os.path.join(output_dir, 'images.txt')
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (pose, img_path) in enumerate(zip(poses, metadata['image_paths'])):
            # Convert transformation matrix to quaternion + translation
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Convert rotation matrix to quaternion
            trace = np.trace(R)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s
            
            # Write image line
            img_name = os.path.basename(img_path)
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
            f.write("\n")  # Empty line for points2D (no 2D-3D correspondences)
    
    # Write empty points3D.txt
    points_file = os.path.join(output_dir, 'points3D.txt')
    with open(points_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    print(f"✅ Converted to COLMAP format in {output_dir}/")
    print(f"   - cameras.txt: Camera intrinsics")
    print(f"   - images.txt: Camera poses")
    print(f"   - points3D.txt: Empty (no sparse reconstruction)")
    
    return output_dir


def visualize_camera_poses(data_path: str):
    """
    Visualize camera poses from transformations.json.
    """
    json_path = os.path.join(data_path, 'transformations.json')
    poses, intrinsics, metadata = load_camera_data(json_path)
    
    # Extract camera positions
    camera_positions = []
    camera_directions = []
    
    for pose in poses:
        # Camera position is translation component
        pos = pose[:3, 3]
        camera_positions.append(pos)
        
        # Camera forward direction is -Z axis in camera frame
        forward = pose[:3, :3] @ np.array([0, 0, -1])
        camera_directions.append(forward)
    
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    
    # Create visualization
    geometries = []
    
    # Add camera positions as spheres
    for pos in camera_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0])  # Red
        geometries.append(sphere)
    
    # Add camera directions as arrows
    for pos, direction in zip(camera_positions, camera_directions):
        end = pos + direction * 0.2
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.15,
            cone_height=0.05
        )
        # Rotate arrow to point in direction
        arrow.translate(pos)
        geometries.append(arrow)
    
    print(f"📹 Visualizing {len(camera_positions)} camera poses...")
    o3d.visualization.draw_geometries(geometries)


def prepare_for_nerf_training(data_path: str, output_dir: str = "nerf_data"):
    """
    Prepare data directory for NeRF training.
    
    Creates a clean directory structure with:
    - images/: All input images
    - transforms_train.json: Training camera poses (nerfstudio format)
    """
    json_path = os.path.join(data_path, 'transformations.json')
    poses, intrinsics, metadata = load_camera_data(json_path)
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy images (or create symlinks)
    import shutil
    base_path = os.path.dirname(json_path)
    
    frames_data = []
    for i, (pose, K_dict, img_path) in enumerate(zip(poses, intrinsics, metadata['image_paths'])):
        # Copy image
        src = os.path.join(base_path, img_path)
        dst = os.path.join(images_dir, f"frame_{i:05d}.jpg")
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"⚠️  Warning: Image not found: {src}")
            continue
        
        # Prepare frame data
        frame_data = {
            "file_path": f"./images/frame_{i:05d}.jpg",
            "transform_matrix": pose.tolist(),
            "fl_x": K_dict['fx'],
            "fl_y": K_dict['fy'],
            "cx": K_dict['cx'],
            "cy": K_dict['cy'],
            "w": K_dict['width'],
            "h": K_dict['height']
        }
        frames_data.append(frame_data)
    
    # Write transforms.json
    transforms_data = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": frames_data
    }
    
    transforms_path = os.path.join(output_dir, 'transforms.json')
    with open(transforms_path, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"✅ Prepared NeRF training data in {output_dir}/")
    print(f"   - {len(frames_data)} images")
    print(f"   - transforms.json: Camera poses and intrinsics")
    print(f"\n📝 Next step: Train NeRF with:")
    print(f"   python nerf_reconstruction.py --data_path {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeRF Data Preparation Utilities")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to directory containing transformations.json")
    parser.add_argument("--action", type=str, 
                       choices=["visualize", "convert_colmap", "prepare"],
                       default="prepare",
                       help="Action to perform")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.action == "visualize":
        visualize_camera_poses(args.data_path)
    elif args.action == "convert_colmap":
        output = args.output_dir or "colmap_format"
        convert_to_colmap_format(args.data_path, output)
    elif args.action == "prepare":
        output = args.output_dir or "nerf_data"
        prepare_for_nerf_training(args.data_path, output)
