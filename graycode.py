'''
This code creates a set of black-and-white stripe images that the projector would shine onto the room.

Example:

Pattern 1 → 01010101…
Pattern 2 → 00110011…
Pattern 3 → 00001111…

Each pattern shifts the stripes to encode position information.

Why this matters:

These stripes help you determine which projector pixel lights up which part of the room.

This is essential for depth sensing.

Real-world analogy:

Imagine shining a barcode onto the room.
By reading the distorted barcode from the camera, you can tell how far each part of the room is.
'''

import numpy as np
import cv2

def generate_gray_code_patterns(width=1024, height=768, num_bits=10):
    patterns = []
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray = x ^ (x >> 1)      # gray code
            bit_value = (gray >> bit) & 1
            pattern[:, x] = 255 * bit_value
        patterns.append(pattern)
    return patterns

patterns = generate_gray_code_patterns()
print("Number of patterns:", len(patterns))
