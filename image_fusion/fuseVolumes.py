import os
import sys

import numpy as np

import nrrd

import numba_interpolate

c_interpolate_seams = True # If yes, cut overlaps between segments to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = True # If yes, apply intensity correction along overlap
c_max_overlap = 4 # Used in interpolation, any segment overlaps are cut to be most this many voxels in size

c_crop_transverse = 3 # Number of transverse slices originally cropped from images before segmentation


def getStationVolumes(path_volumes, path_slices, tag):

    (img, header) = nrrd.read(path_volumes + tag + "_W.nrrd")
    (gt, out) = loadSlices(path_slices, tag)

    gt = reshapeSeg(gt, img.shape)
    out = reshapeSeg(out, img.shape)

    return (img, header, gt, out)


def reshapeSeg(vol, target_shape):

    # Remove padding of segmentation
    shape_dif = np.array(vol.shape) - np.array(target_shape)

    offset = (shape_dif / 2).astype("int")
    vol_out = np.zeros(target_shape)

    start_x = offset[0]
    end_x = offset[0] + target_shape[0]

    start_y = offset[1]
    end_y = offset[1] + target_shape[1]

    c = c_crop_transverse
    vol_out[:, :, c:-c] = vol[start_x:end_x, start_y:end_y, :]

    vol_out = np.flip(vol_out, axis=2)

    return vol_out


def loadSlices(input_path, tag):

    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    files = [f for f in files if tag in f]
    files = [f for f in files if "out.npy" in f]

    station_slices = sorted([f for f in files if tag in f], reverse=True)

    Z = len(station_slices)

    vol_gt = None
    vol_out = None

    #
    for z in range(Z):

        path_z = input_path + station_slices[z]

        slice_out = np.load(path_z)
        slice_gt = np.load(path_z.replace("out.npy", "gt.npy"))

        if vol_gt is None:
            vol_gt = np.zeros((slice_gt.shape[0], slice_gt.shape[1], Z))
            vol_out = np.zeros((slice_gt.shape[0], slice_gt.shape[1], Z))

        vol_out[:, :, z] = slice_out
        vol_gt[:, :, z] = slice_gt

    vol_out = np.swapaxes(vol_out, 0, 1)
    vol_gt = np.swapaxes(vol_gt, 0, 1)

    return (vol_gt, vol_out)


def normalize(values):
    
    if len(np.unique(values)) > 1:
        values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))

    return values


##
# Return, for S segments:
# R:     segment start coordinates, shape Sx3
# R_end: segment end coordinates,   shape Sx3
# dims:  segment extents,           shape Sx3
# 
# Coordinates in R and R_end are in the voxel space of the first segment
def getReadCoordinates(voxel_data, positions, pixel_spacings):

    S = len(voxel_data)

    # Convert from list to arrays
    positions = np.array(positions)
    pixel_spacings = np.array(pixel_spacings)

    # Get dimensions of segments
    dims = np.zeros((S, 3))
    for i in range(S):
        dims[i, :] = voxel_data[i].shape

    # Get segment start coordinates
    R = positions
    origin = np.array(R[0])
    for i in range(S):
        R[i, :] = (R[i, :] - origin) / pixel_spacings[0]

    R[:, 0] -= np.amin(R[:, 0])
    R[:, 1] -= np.amin(R[:, 1])
    R[:, 2] *= -1

    R[:, [0, 1]] = R[:, [1, 0]]

    # Get segment end coordinates
    R_end = np.array(R)
    for i in range(S):
        R_end[i, :] += dims[i, :] * pixel_spacings[i, :] / pixel_spacings[0]

    return (R, R_end, dims)

##
# Ensure that the segments i and (i + 1) overlap by at most c_max_overlap.
# Trim any excess symmetrically
# Update their extents in W and W_end
def trimSegmentOverlaps(W, W_end, W_size, voxel_data):

    W = np.array(W)
    W_end = np.array(W_end)
    W_size = np.array(W_size)

    S = len(voxel_data)
    overlaps = np.zeros(S).astype("int")

    for i in range(S - 1):
        # Get overlap between current and next segment
        overlap = W_end[i, 2] - W[i + 1, 2]

        # No overlap
        if overlap <= 0:
            print("WARNING: No overlap between segments {} and {}. Image might be faulty.".format(i, i+1))

            # Small overlap which can for interpolation
        elif overlap <= c_max_overlap and c_interpolate_seams:
            print("WARNING: Overlap between segments {} and {} is only {}. Using this overlap for interpolation".format(i, i+1, overlap))

        else:
            if c_interpolate_seams:
                cut_a = (overlap - c_max_overlap) / 2.
                overlap = c_max_overlap
            else:
                # Cut at center of seam
                cut_a = overlap / 2.
                overlap = 0

            cut_b = int(np.ceil(cut_a))
            cut_a = int(np.floor(cut_a))

            voxel_data[i] = voxel_data[i][:, :, 0:(W_size[i, 2] - cut_a)]
            voxel_data[i + 1] = voxel_data[i + 1][:, :, cut_b:]

            #
            W_end[i, 2] = W_end[i, 2] - cut_a
            W_size[i, 2] -= cut_a

            W[i + 1, 2] = W[i + 1, 2] + cut_b
            W_size[i + 1, 2] -= cut_b

        overlaps[i] = overlap

    return (overlaps, W, W_end, W_size, voxel_data)


##
# Linearly taper off voxel values along overlap of two segments, 
# so that their addition leads to a linear interpolation.
def fadeSegmentEdges(overlaps, W_size, voxel_data):

    S = len(voxel_data)

    for i in range(S):

        # Only fade inwards facing edges for outer segments
        fadeToPrev = (i > 0)
        fadeToNext = (i < (S - 1))

        # Fade ending edge (facing to next segment)
        if fadeToNext:

            for j in range(overlaps[i]):
                factor = (j+1) / (float(overlaps[i]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, W_size[i, 2] - 1 - j] *= factor

        # Fade starting edge (facing to previous segment)
        if fadeToPrev:

            for j in range(overlaps[i-1]):
                factor = (j+1) / (float(overlaps[i-1]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, j] *= factor

    return voxel_data

## 
# Take mean intensity of slices at the edge of the overlap between segments i and (i+1)
# Adjust mean intensity of each slice along the overlap to linear gradient between these means
def correctOverlapIntensity(overlaps, W_size, voxel_data):

    S = len(voxel_data)

    for i in range(S - 1):
        overlap = overlaps[i]

        # Get average intensity at outer ends of overlap
        edge_a = voxel_data[i+1][:, :, overlap]
        edge_b = voxel_data[i][:, :, W_size[i, 2] - 1 - overlap]

        mean_a = np.mean(edge_a)
        mean_b = np.mean(edge_b)

        for j in range(overlap):

            # Get desired mean intensity along gradient
            factor = (j+1) / (float(overlap) + 1)
            target_mean = mean_b + (mean_a - mean_b) * factor

            # Get current mean of slice when both segments are summed
            slice_b = voxel_data[i][:, :, W_size[i, 2] - overlap + j]
            slice_a = voxel_data[i+1][:, :, j]

            slice_mean = np.mean(slice_a) + np.mean(slice_b)

            # Get correction factor
            correct = target_mean / slice_mean

            voxel_data[i][:, :, W_size[i, 2] - overlap + j] *= correct
            voxel_data[i+1][:, :, j] *= correct

    return voxel_data


def getResamplingParameters(volumes, headers):

    pos_1 = headers[0]["space origin"]
    pos_2 = headers[1]["space origin"]

    spacing = headers[0]["space directions"]
    spacing = np.array((spacing[0][0], spacing[1][1], spacing[2][2]))

    # Determine read coordinates
    (R, R_end, dims) = getReadCoordinates([volumes[0], volumes[1]], [pos_1, pos_2], [spacing, spacing])

    # Determine write coordinates
    W = np.around(R).astype("int")
    W_end = np.around(R_end).astype("int")
    W_size = W_end - W

    #
    scalings = (R_end - R) / dims
    offsets = R - W

    return (W, W_size, W_end, scalings, offsets)


def fuseVolumesRated(volumes_img, headers, volumes_out):

    #
    img_1 = volumes_img[0]
    img_2 = volumes_img[1]

    header_1 = headers[0]
    header_2 = headers[1]

    out_1 = volumes_out[0]
    out_2 = volumes_out[1]

    # 
    (W, W_size, W_end, scalings, offsets) = getResamplingParameters([img_1, img_2], [header_1, header_2])

    #
    (img, img_fusion_cost) = fuseStations(img_1, img_2, W, W_size, W_end, scalings, offsets, True)
    (out, seg_fusion_cost) = fuseStations(out_1, out_2, W, W_size, W_end, scalings, offsets, False)

    header = header_1
    header["sizes"] = img.shape

    return (img, header, out, img_fusion_cost, seg_fusion_cost)


def fuseStations(vol_1, vol_2, W, W_size, W_end, scalings, offsets, is_img):

    # Flip along z axis
    vol_1 = np.ascontiguousarray(np.flip(vol_1, axis=2))
    vol_2 = np.ascontiguousarray(np.flip(vol_2, axis=2))

    #
    vol_2 = numba_interpolate.interpolate3d(W_size[1, :], vol_2, scalings[1, :], offsets[1, :])

    #
    dim_out = np.amax(W_end, axis=0).astype("int")
    vol = np.zeros(dim_out)

    # Trim overlaps
    (overlaps, W, W_end, W_size, [vol_1, vol_2]) = trimSegmentOverlaps(W, W_end, W_size, [vol_1, vol_2])

    fusion_cost = getFusionCost(vol_1, vol_2, overlaps)

    # Taper off segment edges linearly for later addition
    if c_interpolate_seams:
        [vol_1, vol_2] = fadeSegmentEdges(overlaps, W_size, [vol_1, vol_2])

        if not is_img:  
            vol_1 = np.around(vol_1)
            vol_2 = np.around(vol_2)

    # Adjust mean intensity of overlapping slices
    if c_correct_intensity and is_img:
        [vol_1, vol_2] = correctOverlapIntensity(overlaps, W_size, [vol_1, vol_2])

    # Combine
    vol[0:W_end[0, 0], 0:W_end[0, 1], 0:W_end[0, 2]] += vol_1
    vol[W[1, 0]:W_end[1, 0], W[1, 1]:W_end[1, 1], W[1, 2]:W_end[1, 2]] += vol_2

    # Flip back along z axis
    vol = np.ascontiguousarray(np.flip(vol, axis=2))

    return (vol, fusion_cost)


def getFusionCost(vol_1, vol_2, overlaps):

    # Volumes are not yet flipped, so indexing is inverted
    z = overlaps[0]
    fusion_cost = np.sum(np.abs(vol_2.astype("float")[:, :, :z] - vol_1.astype("float")[:, :, -z:])) / z

    # If intensity image, normalize dif by range
    maximum = max((np.amax(vol_1), np.amax(vol_2)))
    minimum = min((np.amin(vol_1), np.amin(vol_2)))
    if maximum > 1:
        fusion_cost = fusion_cost / (maximum - minimum)

    return fusion_cost 


def normalize(values):
    
    if len(np.unique(values)) > 1:
        values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))

    return values
