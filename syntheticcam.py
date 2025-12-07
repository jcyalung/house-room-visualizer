'''
What it does:

This code defines simplified mathematical versions of:

the camera lens

the projector lens

their positions in space

In computer vision, these are called intrinsics and extrinsics.

Why this matters:

To compute 3D points, you need to know how the camera sees the world and where the projector is located relative to the camera.

Real-world analogy:

It’s like knowing:

the size of your camera’s sensor

its focal length

exactly where your projector sits on the table

Without this, depth calculation is impossible.
'''

import numpy as np

# Intrinsics (simple pinhole model)
fx = fy = 800
cx = cy = 400
K_cam  = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]])

K_proj = K_cam.copy()

R_flip = np.array([
    [-1, 0,  0],
    [ 0, 1,  0],
    [ 0, 0, -1]
])

R_cam = R_flip
t_cam = np.array([[0], [0], [2.5]])   # camera 2.5m away from room

# Projector slightly to the right of the camera
R_proj = R_flip
t_proj = np.array([[0.7], [0], [2.5]])
