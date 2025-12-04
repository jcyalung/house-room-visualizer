import numpy as np
import open3d as o3d

def create_cardboard_room(size=1.0, resolution=50):
    """
    Creates a simple 3D 'cardboard room' made of 6 planes (walls).
    size: half-length of the cube (1.0 → cube from -1 to +1)
    resolution: number of samples per dimension (higher → more points)
    """
    points = []

    # Coordinates from -size to +size
    lin = np.linspace(-size, size, resolution)

    # ---- FRONT WALL (z = size) ----
    for x in lin:
        for y in lin:
            points.append([x, y, size])

    # ---- BACK WALL (z = -size) ----
    for x in lin:
        for y in lin:
            points.append([x, y, -size])

    # ---- LEFT WALL (x = -size) ----
    for z in lin:
        for y in lin:
            points.append([-size, y, z])

    # ---- RIGHT WALL (x = size) ----
    for z in lin:
        for y in lin:
            points.append([size, y, z])

    # ---- FLOOR (y = -size) ----
    for x in lin:
        for z in lin:
            points.append([x, -size, z])

    # ---- CEILING (y = size) ----
    for x in lin:
        for z in lin:
            points.append([x, size, z])

    return np.array(points)


def visualize_room(points):
    """
    Visualizes the synthetic room using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("Number of 3D points in room:", len(points))
    o3d.visualization.draw_geometries([pcd])


# ----- MAIN -----
if __name__ == "__main__":
    room_points = create_cardboard_room(size=1.0, resolution=50)
    visualize_room(room_points)
