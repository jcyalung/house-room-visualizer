import numpy as np

def project_points(points, K, R, t):
    """
    Projects 3D points into a camera or projector image plane.

    points: Nx3 array of 3D coordinates
    K: intrinsic matrix
    R: rotation matrix (3x3)
    t: translation vector (3x1)

    Returns: Nx2 array of pixel coordinates
    """
    # Convert points to shape (3, N)
    pts = points.T  

    # Transform into camera/projector coordinate frame
    pts_cam = R @ pts + t  

    # Perspective projection (divide by z)
    pts_norm = pts_cam / pts_cam[2]

    # Apply intrinsics
    pixels = K @ pts_norm

    # Return Nx2
    return pixels[:2].T
