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

# Camera intrinsics
K_cam = np.array([
    [800, 0, 512],
    [0, 800, 384],
    [0, 0, 1]
])

# Projector intrinsics
K_proj = np.array([
    [1000, 0, 512],
    [0, 1000, 384],
    [0, 0, 1]
])

# Camera at origin
R_cam = np.eye(3)
t_cam = np.zeros((3,1))

# Projector shifted sideways
R_proj = np.eye(3)
t_proj = np.array([[0.3, 0, 0]]).T
