import numpy as np

def create_cardboard_room():
    """
    Returns a point cloud representing only the 6 surfaces of a cube:
    - left wall
    - right wall
    - front wall
    - back wall
    - floor
    - ceiling
    """
    size = 2.0  # 2×2×2 meter room
    resolution = 120  # number of points per wall

    lin = np.linspace(-size/2, size/2, resolution)

    walls = []

    # Floor (y = -1)
    X, Z = np.meshgrid(lin, lin)
    Y = np.full_like(X, -size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Ceiling (y = +1)
    Y = np.full_like(X, size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Front wall (z = +1)
    X, Y = np.meshgrid(lin, lin)
    Z = np.full_like(X, size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Back wall (z = -1)
    Z = np.full_like(X, -size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Left wall (x = -1)
    Y, Z = np.meshgrid(lin, lin)
    X = np.full_like(Y, -size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Right wall (x = +1)
    X = np.full_like(Y, size/2)
    walls.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    room = np.vstack(walls)
    return room
