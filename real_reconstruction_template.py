import numpy as np
import open3d as o3d
from triangulation import triangulate

# -------------------------
# STEP 1 — Load calibration
# -------------------------

# TODO: Your friend must insert REAL intrinsics/extrinsics here
K_cam = np.eye(3)
K_proj = np.eye(3)
R_cam = np.eye(3)
t_cam = np.zeros((3,1))
R_proj = np.eye(3)
t_proj = np.zeros((3,1))

# -------------------------
# STEP 2 — Decode Gray-code
# -------------------------

# TODO: Replace this with real function that returns:
# cam_pixels: Nx2 array (camera pixel coords)
# proj_pixels: Nx2 array (projector pixel coords)
def decode_graycode_dataset(path):
    raise NotImplementedError("Insert real Gray-code decoder here.")

cam_pixels, proj_pixels = decode_graycode_dataset("./coffee_room_dataset")

# -------------------------
# STEP 3 — Triangulate points
# -------------------------

points = []
for (cx, cy), (px, py) in zip(cam_pixels, proj_pixels):
    P = triangulate(
        np.array([cx, cy]),
        np.array([px, py]),
        K_cam, K_proj,
        R_cam, t_cam,
        R_proj, t_proj
    )
    points.append(P)

points = np.array(points)

# -------------------------
# STEP 4 — Make point cloud
# -------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# -------------------------
# STEP 5 — Cleanup & mesh
# -------------------------
pcd = pcd.voxel_down_sample(0.01)
pcd.estimate_normals()

mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("coffee_room_mesh.ply", mesh)

print("DONE! Saved coffee_room_mesh.ply")
