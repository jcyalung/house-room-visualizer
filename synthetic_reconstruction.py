import numpy as np
import open3d as o3d

from synthetic_room import create_cardboard_room
from syntheticcam import K_cam, K_proj, R_cam, t_cam, R_proj, t_proj
from project_points import project_points
from triangulation import triangulate


# ---------------------------
# Step 1: Create synthetic room
# ---------------------------
room = create_cardboard_room()


# -----------------------------------------------
# Step 2: Project room points into camera/projector
# -----------------------------------------------
cam_pixels = project_points(room, K_cam, R_cam, t_cam)
proj_pixels = project_points(room, K_proj, R_proj, t_proj)

print("Example camera pixel:", cam_pixels[0])
print("Example projector pixel:", proj_pixels[0])


# ----------------------------
# Step 3: Triangulate (3D recon)
# ----------------------------
# ----------------------------
# Step 3: Triangulate only valid visible points
# ----------------------------
reconstructed_points = []

for true_point in room:
    # Project into camera and projector
    cam_pix = project_points(true_point.reshape(1,3), K_cam, R_cam, t_cam)[0]
    proj_pix = project_points(true_point.reshape(1,3), K_proj, R_proj, t_proj)[0]

    # Only keep points inside both sensor FOVs
    if not (0 <= cam_pix[0] < 800 and 0 <= cam_pix[1] < 600):
        continue
    if not (0 <= proj_pix[0] < 800 and 0 <= proj_pix[1] < 600):
        continue

    # Triangulate
    P = triangulate(cam_pix, proj_pix,
                    K_cam, K_proj,
                    R_cam, t_cam,
                    R_proj, t_proj)

    reconstructed_points.append(P)

reconstructed_points = np.array(reconstructed_points)
print("Reconstructed points:", reconstructed_points.shape)



# -----------------------------------
# Step 4: Create point cloud in Open3D
# -----------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(reconstructed_points)


# -----------------------------------
# Step 5: Clean the point cloud
# -----------------------------------
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)

# TEMP: Do not remove outliers — synthetic cube has no noise
pcd_clean = pcd_down

print("Cleaned point cloud size:", np.asarray(pcd_clean.points).shape)
o3d.visualization.draw_geometries([pcd_clean])

