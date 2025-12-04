'''
What it does:

This code takes the sequence of stripes (from the camera’s point of view) and decodes them into a projector coordinate (like column number 0–1023).

Why this matters:

The camera sees an object.
The projector also “knows” where it projected each stripe.

When you match them, you get:

camera pixel  →  projector pixel


This correspondence is what allows 3D reconstruction.

Real-world analogy:

If two people look at the same object from two sides, and they both point at the same spot, you can figure out the 3D position of that spot.
'''


def gray_to_binary(gray):
    binary = gray
    shift = 1
    while (gray >> shift) > 0:
        binary ^= (gray >> shift)
        shift += 1
    return binary

def decode_gray_sequence(sequence):
    # sequence: list of 0/1 values for each bit
    gray_value = 0
    for i, bit in enumerate(sequence):
        gray_value |= (bit << i)
    binary_value = gray_to_binary(gray_value)
    return binary_value
