import sys
import os

import numpy as np

import time

import nrrd
import torch
from torch.autograd import Variable

import dicomToVolume
import fuseVolumes

# Target size for input samples
c_x = 224
c_y = 192

# Number of axial slices to be removed from top and bottom of output to avoid segmenting folding artefacts
c_crop_axial = 3 


#####
# Given two imaging stations with water signal for a given subject
# Make slice-wise predictions and fuse both images and resulting binary segmentation
def predictForSubject(station_vols, headers, net):

    # Make prediction with network
    output_vols = predict(net, station_vols)

    # Fuse and calculate fusion cost terms
    (img, header, out, img_fusion_cost, seg_fusion_cost) = fuseVolumes.fuseVolumesRated(station_vols, headers, output_vols)

    # Remove any segmentation in outermost slices, which may contain folding artefacts
    out[:, :, :c_crop_axial] = 0
    out[:, :, -c_crop_axial:] = 0

    return (img, out, header, img_fusion_cost, seg_fusion_cost)


def predict(net, station_vols):

    net.eval()

    output_vols = []

    for i in range(len(station_vols)):

        # Get input slices from station
        X = formatInputSlices(station_vols[i])
        X = X.cuda(non_blocking=True)

        # Predict
        output = net(X).detach()
        del X   
        output_vol = np.argmax(output.cpu().data.numpy()[:, :, :, :], 1)

        # Format
        output_vol = np.swapaxes(output_vol, 0, 2)
        output_vol = reshapeSeg(output_vol, station_vols[i].shape)
        output_vols.append(output_vol)

    return output_vols


def reshapeSeg(vol, target_shape):

    # Revert padding
    shape_dif = np.array(vol.shape) - np.array(target_shape)

    offset = (shape_dif / 2).astype("int")
    vol_out = np.zeros(target_shape)

    start_x = offset[0]
    end_x = offset[0] + target_shape[0]

    start_y = offset[1]
    end_y = offset[1] + target_shape[1]

    vol_out[:, :, :] = vol[start_x:end_x, start_y:end_y, :]

    return vol_out


# Extract 2.5D samples from input volume
def formatInputSlices(input_vol):

    Z = input_vol.shape[2]
    slices_out = np.zeros((Z, 3, c_y, c_x))

    # Normalize contrast, clipping brightest 1% of intensities
    input_vol = normalizeClipVolume(input_vol)

    # Pad to target size along axial plane
    target_shape = np.array((c_x, c_y))
    pad = ((target_shape - input_vol.shape[:2]) / 2).astype("int")

    input_vol = np.pad(input_vol, 
        (
            (pad[0], pad[0]), 
            (pad[1], pad[1]), 
            (0, 0)
        ), 
        mode="constant", constant_values = 0)

    # For each axial slice, form a 2.5D representation of adjacent slices
    for z in range(Z):

        # Extract axial slices with periodic border condition
        z_range = np.clip(np.arange(-1, 2) + z, 0, Z-1)
        slices_out[z, :, :, :] = np.swapaxes(input_vol[:, :, z_range], 0, 2)

    #
    X = Variable(torch.from_numpy(slices_out)).float()

    return X


def normalizeClipVolume(vol):

    vol = np.copy(vol)

    Z = vol.shape[2]

    # For each axial slice, normalize intensity values 
    # after clipping of brightest 1%
    for z in range(Z):

        values = vol[:, :, z]
    
        # Clip brightest 1%
        t = np.ceil(np.percentile(values, 99))
        values[values > t] = t
        
        # Normalize
        if len(np.unique(values)) > 1:
            values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
        else:
            values[:] = 0

        vol[:, :, z] = values

    return vol
