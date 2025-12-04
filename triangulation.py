'''
What it does:

This is the heart of your 3D reconstruction.

Given:

a pixel in the camera image

the matching pixel in the projector

the geometry of how both are positioned in space

It computes the 3D point in space that caused that pixel.

This is literally how you get the point cloud.

Why this matters:

This code converts:

2D pixels → 3D coordinates


which is the entire goal of the structured-light method.

Real-world analogy:

If two people look at the same object from different angles and point toward it, the intersection of their lines of sight gives you the exact location of the object.

Triangulation = intersecting the lines of sight.
'''


def triangulate(cam_pixel, proj_pixel, K_cam, K_proj, R_cam, t_cam, R_proj, t_proj):
    # backproject rays from camera and projector
    cam_dir = np.linalg.inv(K_cam) @ np.array([cam_pixel[0], cam_pixel[1], 1])
    proj_dir = np.linalg.inv(K_proj) @ np.array([proj_pixel[0], proj_pixel[1], 1])

    cam_ori = -np.linalg.inv(R_cam) @ t_cam
    proj_ori = -np.linalg.inv(R_proj) @ t_proj

    # linear triangulation
    A = np.stack([cam_dir, -proj_dir], axis=1)
    b = proj_ori.flatten() - cam_ori.flatten()

    lambdas, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    P = cam_ori.flatten() + lambdas[0] * cam_dir
    return P
