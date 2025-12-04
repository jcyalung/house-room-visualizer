# house-room-visualizer
Written by Joshua Yalung and Jerome Manarang.

Visualizer for a house room using neural radiance fields.

Designed for CS117: Project in Computer Vision. 


Dec 3:
Our project currently includes the foundational components of the structured-light 3D reconstruction pipeline. Although we have not yet captured any physical data, we have fully implemented the core algorithms required for projector–camera based depth reconstruction. These components will later integrate directly with real cardboard-room captures when physical setup becomes available.

1. Gray-Code Pattern Generation

We implemented a complete Gray-code pattern generator that produces the binary structured-light patterns used by a projector.
These patterns uniquely encode projector column indices and are essential for establishing projector–camera pixel correspondences during reconstruction.

Status: ✔️ Fully implemented
Input: Image resolution (width/height), number of bits
Output: Set of binary stripe images (patterns)

2. Gray-Code Decoding

We implemented the decoding logic that converts captured binary patterns back into projector coordinates.
Given a sequence of bit values for a single pixel, the decoder reconstructs the original Gray-code value and converts it into a binary index.

This serves as the core of the correspondence estimation process.

Status: ✔️ Fully implemented
Input: Bit sequence for a camera pixel
Output: Projector pixel index (column)

3. Synthetic Camera and Projector Models

We constructed mathematical models of the camera and projector, including pinhole intrinsics (focal length, principal point) and extrinsics (position and orientation in 3D space).
This synthetic setup allows us to develop and test the full reconstruction pipeline before collecting physical data.

Status: ✔️ Fully implemented
Input: Defined intrinsic matrices and 3D transformations
Output: Virtual camera/projector configuration used for simulation and triangulation

4. 3D Triangulation Framework

We implemented a linear triangulation algorithm that recovers the 3D position of a point given its corresponding pixel locations in the camera and projector.
This is the key step that transforms 2D structured-light data into a 3D point cloud.

Status: ✔️ Fully implemented
Input: (camera pixel, projector pixel), camera + projector calibration
Output: 3D point in space