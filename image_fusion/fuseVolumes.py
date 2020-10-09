import os
import sys
import io

import time


import numpy as np

import scipy.interpolate
import numba_interpolate

from skimage import filters

import nrrd
import cv2

c_resample_tolerance = 0.01 # Only interpolate voxels further off of the voxel grid than this

c_interpolate_seams = True # If yes, cut overlaps between stations to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = True # If yes, apply intensity correction along overlap
c_max_overlap = 4 # Used in interpolation, any station overlaps are cut to be most this many voxels in size

c_trim_axial_slices = 4 # Trim this many axial slices from the output volume to remove folding artefacts

c_use_gpu = True # If yes, use numba for gpu access, otherwise use scipy on cpu


#
def fuseStations(voxels, positions, pixel_spacings, target_spacing, is_img):

    for i in range(len(voxels)): 
        # Flip along z axis
        voxels[i] = np.ascontiguousarray(np.flip(voxels[i], axis=2))

    # Resample stations onto output volume voxel grid
    (voxels, W, W_end, W_size, shifts) = resampleStations(voxels, positions, pixel_spacings, target_spacing)

    # Cut station overlaps to at most c_max_overlap
    (overlaps, W, W_end, W_size, voxels) = trimStationOverlaps(W, W_end, W_size, voxels)

    # Combine stations to volumes
    (volume, fusion_cost) = fuseVolume(W, W_end, W_size, voxels, overlaps, is_img) 

    # Flip back along z axis
    volume = np.ascontiguousarray(np.swapaxes(volume, 0, 1))

    origin = positions[-1] + shifts[-1, :]

    if not is_img:
        volume = np.around(volume)

    return (volume, origin, fusion_cost)


# Save volumetric nrrd file
def storeNrrd(volume, output_path, origin):

    # See: http://teem.sourceforge.net/nrrd/format.html
    header = {'dimension': 3}
    header['type'] = "float"
    header['sizes'] = volume.shape

    # Spacing info compatible with 3D Slicer
    header['space dimension'] = 3
    header['space directions'] = np.array(target_spacing * np.eye(3,3))
    header['space origin'] = origin
    header['space units'] = "\"mm\" \"mm\" \"mm\""
    header['encoding'] = 'gzip'

    #
    nrrd.write(output_path + ".nrrd", volume, header, compression_level=1)


# Generate mean intensity projection 
def formatMip(volume):

    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]

    # Coronal projection
    slice_cor = np.sum(volume, axis = 1)
    slice_cor = np.rot90(slice_cor, 1)

    # Sagittal projection
    slice_sag = np.sum(volume, axis = 0)
    slice_sag = np.rot90(slice_sag, 1)

    # Normalize intensities
    slice_cor = (normalize(slice_cor) * 255).astype("uint8")
    slice_sag = (normalize(slice_sag) * 255).astype("uint8")

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)
    slice_out = cv2.resize(slice_out, (256, 256))

    return slice_out


def normalize(img):

    img = img.astype("float")
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return img


##
# Form sum of absolute differences between station overlaps
# Normalize by overlap size and intensity range (if image)
def getFusionCost(W, W_end, voxels, overlaps, is_img):

    S = len(voxels)

    cost = 0
    for i in range(S-1):

        # Get coordinates pointing to spatially corresponding voxels in both stations
        start_0 = np.clip(W[i+1]-W[i], 0, None)
        start_1 = np.clip(W[i]-W[i+1], 0, None)

        end_0 = voxels[i].shape - np.clip(W_end[i] - W_end[i+1] , 0, None)
        end_1 = voxels[i+1].shape - np.clip(W_end[i+1] - W_end[i], 0, None)

        # Get difference in overlap
        dif_i = voxels[i][start_0[0]:end_0[0], start_0[1]:end_0[1], -overlaps[i]:] - voxels[i+1][start_1[0]:end_1[0], start_1[1]:end_1[1], :overlaps[i]]

        # Form sum of absolute differences, normalized by intensity range and overlap size
        dif_i = np.sum(np.abs(dif_i)) / overlaps[i]

        if is_img:

            # For signal images, normalize fusion cost with intensity range of involved stations
            max_i = max(np.amax(voxels[i]), np.amax(voxels[i+1]))
            min_i = min(np.amin(voxels[i]), np.amin(voxels[i+1]))

            dif_i = dif_i / (max_i - min_i)

        cost += dif_i

    return cost


def fuseVolume(W, W_end, W_size, voxels, overlaps, is_img):

    S = len(voxels)

    # Cast to datatype
    for i in range(S):  
        voxels[i] = voxels[i].astype("float32")

    # 
    fusion_cost = getFusionCost(W, W_end, voxels, overlaps, is_img)

    # Taper off station edges linearly for later addition
    if c_interpolate_seams:
        voxels = fadeStationEdges(overlaps, W_size, voxels)

        #if not is_img:  
            #for i in range(S): voxels[i] = np.around(voxels[i])

    # Adjust mean intensity of overlapping slices
    if is_img and c_correct_intensity:
        voxels = correctOverlapIntensity(overlaps, W_size, voxels)

    # Combine stations into volume by addition
    volume = combineStationsToVolume(W, W_end, voxels)

    if False:
        # Remove slices affected by folding
        if c_trim_axial_slices > 0:
            start = c_trim_axial_slices
            end = volume.shape[2] - c_trim_axial_slices
            volume = volume[:, :, start:end]

    return (volume, fusion_cost)


def combineStationsToVolume(W, W_end, voxels):

    S = len(voxels)

    volume_dim = np.amax(W_end, axis=0).astype("int")
    volume = np.zeros(volume_dim)

    for i in range(S):
        volume[W[i, 0]:W_end[i, 0], W[i, 1]:W_end[i, 1], W[i, 2]:W_end[i, 2]] += voxels[i][:, :, :]

    #
    volume = np.flip(volume, 2)
    volume = np.swapaxes(volume, 0, 1)

    return volume


##
# Return, for S stations:
# R:     station start coordinates, shape Sx3
# R_end: station end coordinates,   shape Sx3
# dims:  station extents,           shape Sx3
# 
# Coordinates in R and R_end are in the voxel space of the first station
def getReadCoordinates(voxels, positions, pixel_spacings, target_spacing):

    S = len(voxels)

    # Convert from list to arrays
    positions = np.array(positions)
    pixel_spacings = np.array(pixel_spacings)

    # Get dimensions of stations
    dims = np.zeros((S, 3))
    for i in range(S):
        dims[i, :] = voxels[i].shape

    # Get station start coordinates
    R = positions
    origin = np.array(R[0])
    for i in range(S):
        R[i, :] = (R[i, :] - origin) / target_spacing

    R[:, 0] -= np.amin(R[:, 0])
    R[:, 1] -= np.amin(R[:, 1])
    R[:, 2] *= -1

    R[:, [0, 1]] = R[:, [1, 0]]

    # Get station end coordinates
    R_end = np.array(R)
    for i in range(S):
        R_end[i, :] += dims[i, :] * pixel_spacings[i, :] / target_spacing

    return (R, R_end, dims)


##
# Linearly taper off voxel values along overlap of two stations, 
# so that their addition leads to a linear interpolation.
def fadeStationEdges(overlaps, W_size, voxels):

    S = len(voxels)

    for i in range(S):

        # Only fade inwards facing edges for outer stations
        fadeToPrev = (i > 0)
        fadeToNext = (i < (S - 1))

        # Fade ending edge (facing to next station)
        if fadeToNext:

            for j in range(overlaps[i]):
                factor = (j+1) / (float(overlaps[i]) + 1) # exclude 0 and 1
                voxels[i][:, :, W_size[i, 2] - 1 - j] *= factor

        # Fade starting edge (facing to previous station)
        if fadeToPrev:

            for j in range(overlaps[i-1]):
                factor = (j+1) / (float(overlaps[i-1]) + 1) # exclude 0 and 1
                voxels[i][:, :, j] *= factor

    return voxels


## 
# Take mean intensity of slices at the edge of the overlap between stations i and (i+1)
# Adjust mean intensity of each slice along the overlap to linear gradient between these means
def correctOverlapIntensity(overlaps, W_size, voxels):

    S = len(voxels)

    for i in range(S - 1):
        overlap = overlaps[i]

        # Get average intensity at outer ends of overlap
        edge_a = voxels[i+1][:, :, overlap]
        edge_b = voxels[i][:, :, W_size[i, 2] - 1 - overlap]

        mean_a = np.mean(edge_a)
        mean_b = np.mean(edge_b)

        for j in range(overlap):

            # Get desired mean intensity along gradient
            factor = (j+1) / (float(overlap) + 1)
            target_mean = mean_b + (mean_a - mean_b) * factor

            # Get current mean of slice when both stations are summed
            slice_b = voxels[i][:, :, W_size[i, 2] - overlap + j]
            slice_a = voxels[i+1][:, :, j]

            slice_mean = np.mean(slice_a) + np.mean(slice_b)

            # Get correction factor
            correct = target_mean / slice_mean

            # correct intensity to match linear gradient
            voxels[i][:, :, W_size[i, 2] - overlap + j] *= correct
            voxels[i+1][:, :, j] *= correct

    return voxels


##
# Ensure that the stations i and (i + 1) overlap by at most c_max_overlap.
# Trim any excess symmetrically
# Update their extents in W and W_end
def trimStationOverlaps(W, W_end, W_size, voxels):

    W = np.array(W)
    W_end = np.array(W_end)
    W_size = np.array(W_size)

    S = len(voxels)
    overlaps = np.zeros(S).astype("int")

    for i in range(S - 1):
        # Get overlap between current and next station
        overlap = W_end[i, 2] - W[i + 1, 2]

        # No overlap
        if overlap <= 0:
            print("WARNING: No overlap between stations {} and {}. Image might be faulty.".format(i, i+1))

        # Small overlap which can for interpolation
        elif overlap <= c_max_overlap and c_interpolate_seams:
            print("WARNING: Overlap between stations {} and {} is only {}. Using this overlap for interpolation".format(i, i+1, overlap))

        # Large overlap which must be cut
        else:
            if c_interpolate_seams:
                # Keep an overlap of at most c_max_overlap
                cut_a = (overlap - c_max_overlap) / 2.
                overlap = c_max_overlap
            else:
                # Cut at center of seam
                cut_a = overlap / 2.
                overlap = 0

            cut_b = int(np.ceil(cut_a))
            cut_a = int(np.floor(cut_a))

            voxels[i] = voxels[i][:, :, 0:(W_size[i, 2] - cut_a)]
            voxels[i + 1] = voxels[i + 1][:, :, cut_b:]

            #
            W_end[i, 2] = W_end[i, 2] - cut_a
            W_size[i, 2] -= cut_a

            W[i + 1, 2] = W[i + 1, 2] + cut_b
            W_size[i + 1, 2] -= cut_b

        overlaps[i] = overlap

    return (overlaps, W, W_end, W_size, voxels)


##
# Station voxels are positioned at R to R_end, not necessarily aligned with output voxel grid
# Resample stations onto voxel grid of output volume
def resampleStations(voxels, positions, pixel_spacings, target_spacing):

    # R: station positions off grid respective to output volume
    # W: station positions on grid after resampling
    (R, R_end, dims) = getReadCoordinates(voxels, positions, pixel_spacings, target_spacing)

    # Get coordinates of voxels to write to
    W = np.around(R).astype("int")
    W_end = np.around(R_end).astype("int")
    W_size = W_end - W

    shift = (R - W) * pixel_spacings

    result_data = []

    #
    for i in range(len(voxels)):

        # Get largest offset off of voxel grid
        offsets = np.concatenate((R[i, :].flatten(), R_end[i, :].flatten()))
        offsets = np.abs(offsets - np.around(offsets))

        max_offset = np.amax(offsets)

        # Get difference in voxel counts
        voxel_count_out = np.around(W_size[i, :])
        voxel_count_dif = np.sum(voxel_count_out - dims[i, :])

        # No resampling if station voxels are already aligned with output voxel grid
        doResample = (max_offset > c_resample_tolerance or voxel_count_dif != 0)

        result = None
        
        if doResample:

            if c_use_gpu:

                # Use numba implementation on gpu:
                scalings = (R_end[i, :] - R[i, :]) / dims[i, :]
                offsets = R[i, :] - W[i, :] 
                result = numba_interpolate.interpolate3d(W_size[i, :], voxels[i], scalings, offsets)

            else:
                # Use scipy CPU implementation:
                # Define positions of station voxels (off of output volume grid)
                x_s = np.linspace(int(R[i, 0]), int(R_end[i, 0]), int(dims[i, 0]))
                y_s = np.linspace(int(R[i, 1]), int(R_end[i, 1]), int(dims[i, 1]))
                z_s = np.linspace(int(R[i, 2]), int(R_end[i, 2]), int(dims[i, 2]))

                # Define positions of output volume voxel grid
                y_v = np.linspace(W[i, 0], W_end[i, 0], W_size[i, 0])
                x_v = np.linspace(W[i, 1], W_end[i, 1], W_size[i, 1])
                z_v = np.linspace(W[i, 2], W_end[i, 2], W_size[i, 2])

                xx_v, yy_v, zz_v = np.meshgrid(x_v, y_v, z_v)

                pts = np.zeros((xx_v.size, 3))
                pts[:, 1] = xx_v.flatten()
                pts[:, 0] = yy_v.flatten()
                pts[:, 2] = zz_v.flatten()

                # Resample stations onto output voxel grid
                rgi = scipy.interpolate.RegularGridInterpolator((x_s, y_s, z_s), voxels[i], bounds_error=False, fill_value=None)
                result = rgi(pts)

        else:
            # No resampling necessary
            result = voxels[i]

        result_data.append(result.reshape(W_size[i, :]))

    return (result_data, W, W_end, W_size, shift)
