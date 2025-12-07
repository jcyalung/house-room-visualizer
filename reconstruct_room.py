from synthetic_room import create_cardboard_room
from syntheticcam import K_cam, K_proj, R_cam, t_cam, R_proj, t_proj
from project_points import project_points
from triangulation import triangulate
import numpy as np
import open3d as o3d

# Step 1: Build synthetic room
room = create_cardboard_room()

# Step 2: Project into camera + projector
cam_pixels = project_points(room, K_cam, R_cam, t_cam)
proj_pixels = project_points(room, K_proj, R_proj, t_proj)

print("Example camera pixel:", cam_pixels[0])
print("Example projector pixel:", proj_pixels[0])

# Step 3: Reconstruct using triangulation
reconstructed_points = []

for cam_pix, proj_pix in zip(cam_pixels, proj_pixels):
    P = triangulate(cam_pix, proj_pix,
                    K_cam, K_proj,
                    R_cam, t_cam,
                    R_proj, t_proj)
    reconstructed_points.append(P)

reconstructed_points = np.array(reconstructed_points)
print("Reconstruction complete! Shape:", reconstructed_points.shape)

# Step 4: Visualize the reconstructed point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(reconstructed_points)

o3d.visualization.draw_geometries([pcd])


# Clean up the point cloud
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)

pcd_clean, ind = pcd_down.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

print("Cleaned point cloud size:", np.asarray(pcd_clean.points).shape)

o3d.visualization.draw_geometries([pcd_clean])
