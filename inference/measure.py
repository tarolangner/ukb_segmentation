import os
import sys

import shutil
import cv2

import numpy as np
import skimage.measure 
import nrrd


def identifySide(centers):

    # Background and two components, identify
    if centers[1, 0] > centers[2, 0]:
        return (1, 2)
        
    return (2, 1)


def measureKidneys(seg, voxel_dim):

    # Get connected components
    label_img = skimage.measure.label(seg) 

    L = np.amax(label_img) + 1

    label_volumes = np.zeros(L)
    label_coms = np.zeros((L, 3)) # center of mass

    for l in range(L):

        label_volumes[l] = np.sum(label_img == l)

        label_idx = np.where(label_img == l)
        label_coms[l, :] = np.mean(label_idx, axis=1)

    labels = np.arange(L)

    # Sort descending by volume (first will be background)
    idx = np.argsort(label_volumes)[::-1]
    label_volumes = label_volumes[idx]
    label_coms = label_coms[idx, :]
    labels = labels[idx]

    label_mask = np.ones(label_img.shape) * 3 # initialize with scrap marker

    voxel_scale = np.prod(voxel_dim) / (10*10*10)

    # Check if background + 2 components exist
    if L > 2:

        #
        (idx_left, idx_right) = identifySide(label_coms)

        # Mark label mask
        label_mask[label_img == labels[0]] = 0
        label_mask[label_img == labels[idx_left]] = 1
        label_mask[label_img == labels[idx_right]] = 2

        # Get volumes
        volume_left = label_volumes[idx_left] * voxel_scale
        volume_right = label_volumes[idx_right] * voxel_scale
        volume_total = np.sum(label_volumes[1:]) * voxel_scale

        # Get offsets
        offsets = (label_coms[idx_left] - label_coms[idx_right]) * voxel_dim

        # Get average longitudinal position of kidneys
        kidney_z = (label_coms[idx_left][2] + label_coms[idx_right][2]) / 2

    elif L == 1:

        # Only background
        volume_left = 0
        volume_right = 0
        volume_total = 0
        kidney_z = seg.shape[2] / 2
        offsets = np.zeros(3)

        # Mark label mask
        label_mask[label_img == labels[0]] = 0
        
    else:
        label_mask[label_img == labels[0]] = 0

        # Only one kidney, identify by relative position to background
        if label_coms[1, 0] > label_coms[0, 0]:

            # Left kidney only
            volume_left = label_volumes[1] * voxel_scale
            volume_right = 0

            label_mask[label_img == labels[1]] = 1
        else:
            # Right kidney only
            volume_right = label_volumes[1] * voxel_scale
            volume_left = 0

            label_mask[label_img == labels[1]] = 2

        offsets = np.zeros(3)
        volume_total = volume_left + volume_right
        kidney_z = label_coms[1][2]

    #
    kidney_z = 2 * (np.abs(kidney_z - seg.shape[2] / 2) / seg.shape[2])
            
    return (volume_left, volume_right, volume_total, offsets, kidney_z, label_mask)



def measurePancreas(seg, voxel_dim):

    voxel_scale = np.prod(voxel_dim) / (10*10*10)

    volume_total = seg * voxel_scale

    # Centre of mass
    label_idx = np.where(seg == 1)
    label_com = np.mean(label_idx, axis=1)

    z_cost = 2 * (np.abs(label_com[2] - seg.shape[2] / 2) / seg.shape[2])

    return (volume_total, z_cost)
