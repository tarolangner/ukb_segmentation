from __future__ import division
from numba import cuda
import numpy as np
import math

# Interpolation of 3D grids on graphic card

# CUDA kernel
@cuda.jit
def interp3d(A, scalings, offsets, B):
    
    # Fill B with values
    # A contains values for input grid
    # Scalings is size of voxels in A relative to B
    # Offsets is shift of voxels in A relative to B (measured in size of voxels in B)

    x, y, z = cuda.grid(3)

    if x < B.shape[0] and y < B.shape[1] and z < B.shape[2]:

        # Get position in voxel space of A
        ax = (x - offsets[0]) / scalings[0]
        ay = (y - offsets[1]) / scalings[1]
        az = (z - offsets[2]) / scalings[2]

        # Get closest two voxels
        ax0 = int(ax)
        ax1 = ax0 + 1

        ay0 = int(ay)
        ay1 = ay0 + 1

        az0 = int(az)
        az1 = az0 + 1

        # Constrain to image
        if ax >= A.shape[0]: ax = (A.shape[0] - 1)
        if ax0 >= A.shape[0]: ax0 = (A.shape[0] - 1)
        if ax1 >= A.shape[0]: ax1 = (A.shape[0] - 1)

        if ay >= A.shape[1]: ay = (A.shape[1] - 1)
        if ay0 >= A.shape[1]: ay0 = (A.shape[1] - 1)
        if ay1 >= A.shape[1]: ay1 = (A.shape[1] - 1)

        if az >= A.shape[2]: az = (A.shape[2] - 1)
        if az0 >= A.shape[2]: az0 = (A.shape[2] - 1)
        if az1 >= A.shape[2]: az1 = (A.shape[2] - 1)

        if ax < 0 : ax = 0
        if ax0 < 0 : ax0 = 0
        if ax1 < 0 : ax1 = 0

        if ay < 0 : ay = 0
        if ay0 < 0 : ay0 = 0
        if ay1 < 0 : ay1 = 0

        if az < 0 : az = 0
        if az0 < 0 : az0 = 0
        if az1 < 0 : az1 = 0

        # Get normalized distance from previous voxel
        dx = ax - ax0
        dy = ay - ay0
        dz = az - az0

        # Interpolate along x
        cy0z0 = dx * A[ax1, ay0, az0] + (1 - dx) * A[ax0, ay0, az0]
        cy1z0 = dx * A[ax1, ay1, az0] + (1 - dx) * A[ax0, ay1, az0]
        cy0z1 = dx * A[ax1, ay0, az1] + (1 - dx) * A[ax0, ay0, az1]
        cy1z1 = dx * A[ax1, ay1, az1] + (1 - dx) * A[ax0, ay1, az1]

        # Interpolate along y
        cz0 = dy * cy1z0 + (1 - dy) * cy0z0
        cz1 = dy * cy1z1 + (1 - dy) * cy0z1

        # Interpolate along z
        out = dz * cz1 + (1 - dz) * cz0

        B[x, y, z] = out
        
# Host code

def interpolate3d(output_shape, values, scalings, offsets):

    output = np.zeros(output_shape)

    # Copy the arrays to the device
    global_mem_scalings = cuda.to_device(scalings)
    global_mem_offsets = cuda.to_device(offsets)
    global_mem_values = cuda.to_device(values)
    global_mem_output = cuda.to_device(output)

    # Configure the blocks
    threadsperblock = (16, 16, 2)
    blockspergrid_x = int(math.ceil(output.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(output.shape[1] / threadsperblock[1]))
    blockspergrid_z = int(math.ceil(output.shape[2] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Start the kernel 
    interp3d[blockspergrid, threadsperblock](global_mem_values, global_mem_scalings, global_mem_offsets, global_mem_output)

    # Copy the result back to the host
    output = global_mem_output.copy_to_host()

    return output
