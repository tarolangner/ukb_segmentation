import sys
import os

import numpy as np

import time

from torch.utils import data

import shutil
import cv2
import nrrd

import torch
from torch.autograd import Variable

sys.path.insert(0, "../image_fusion/")
sys.path.insert(0, "../cross_validation/scripts/models/")

#
import predictForSubject
import dicomToVolume
import fuseVolumes
import measure

from model_resUnet import UNet

c_write_nrrd = False # Write output volumes in nrrd format
c_write_mip = True # write output as mean intensity projections in png format

c_target_spacing = np.array((2.23214293, 2.23214293, 4.5)) # Target spacing to which all voxels are resampled when fusing stations

def main(argv):

    path_ids = "ids.txt" # List of subject ids to be processed
    path_dicom = "/media/veracrypt1/UKB_DICOM/" # Path to UKB dicoms

    path_checkpoint = "/media/taro/DATA/Taro/Projects/ukb_segmentation/cross-validation/networks/kidney_122_traintest_watRoi192_deform_80kLR/subset_0/snapshots/iteration_080000.pth.tar"
    path_out = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/inference_kidney_122/"

    # Select which MRI stations to perform inference on
    station_ids = [1, 2]

    #
    time_start = time.time()

    ###    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        os.makedirs(path_out + "NRRD/")
        os.makedirs(path_out + "MIP/")
    else:
        print("ABORT: Output folder already exists!")
        sys.exit()

    # Open output measurement file and later keep appending to it
    with open(path_out + "measurements.txt", "w") as f:
        f.write("eid,total_kidney_tissue_in_ml,kidney_left_tissue_in_ml,kidney_right_tissue_in_ml,distance_x_in_mm,distance_y_in_mm,distance_z_in_mm\n")

    # Open quality metric file and later keep appending to it
    with open(path_out + "quality.txt", "w") as f:
        f.write("eid,img_fusion_cost,seg_fusion_cost,seg_smoothness,kidney_z_cost\n")

    # Read ids
    with open(path_ids) as f: entries = f.readlines()
    subject_ids = [f.split(",")[0].replace("\n","") for f in entries]
    subject_ids = np.ascontiguousarray(np.array(subject_ids).astype("int"))

    #
    N = len(subject_ids)
    print("Found {} subject ids...".format(N))

    # 
    print("Initializing network...")
    net = UNet(3, 2).cuda()
    checkpoint = torch.load(path_checkpoint, map_location={"cuda" : "cpu"})
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    # Use pytorch data loader for parallel loading and prediction
    loader = getDataloader(path_dicom, subject_ids, station_ids)

    #
    i = 0
    for station_vols, headers, subject_id in loader:

        subject_id = subject_id[0]
        print("Processing subject {0} ({1:0.3f}% completed)".format(subject_id, 100 * i / N))

        processSubject(station_vols, headers, net, subject_id, path_out)

        i += 1

    # Write runtime
    time_end = time.time()
    runtime = time_end - time_start

    print("Elapsed time: {}".format(runtime))
    with open(path_out + "runtime.txt", "w") as f:
        f.write("{}".format(runtime))


def processSubject(station_vols, headers, net, subject_id, path_out):

    # Revert batch and tensor formatting by dataloader
    # Fite me irl
    for i in range(len(station_vols)):
        station_vols[i] = station_vols[i].data.numpy()[0, :, :, :]

        headers[i]["space origin"] = headers[i]["space origin"].data.numpy()[0, :]
        headers[i]["space directions"] = headers[i]["space directions"].data.numpy()[0, :]
        headers[i]["encoding"] = headers[i]["encoding"][0]
        headers[i]["dimension"] = headers[i]["dimension"].data.numpy()[0]
        headers[i]["space dimension"] = headers[i]["space dimension"].data.numpy()[0]

    # Predict and fuse stations
    (img, out, header, img_fusion_cost, seg_fusion_cost) = predictForSubject.predictForSubject(station_vols, headers, net, c_target_spacing, True)

    # 
    voxel_dim = c_target_spacing

    # Get measurements and quality ratings from output
    (volume_left, volume_right, volume_total, offsets, kidney_z_cost, label_mask) = measure.measureKidneys(out, voxel_dim)
    seg_smoothness = rateSegmentationSmoothness(out)

    # Append to previously opened text files
    writeTxtLine(path_out + "measurements.txt", [subject_id, volume_total, volume_left, volume_right, offsets[0], offsets[1], offsets[2]])
    writeTxtLine(path_out + "quality.txt", [subject_id, img_fusion_cost, seg_fusion_cost, seg_smoothness, kidney_z_cost])

    # Write volumes
    if c_write_nrrd:

        nrrd.write(path_out + "NRRD/{}_img.nrrd".format(subject_id), img, header, compression_level=1)
        nrrd.write(path_out + "NRRD/{}_out.nrrd".format(subject_id), out, header, compression_level=1)

    # Write mean intensity projections
    if c_write_mip:

        proj_out = formatMip(img, out, label_mask)
        cv2.imwrite(path_out + "MIP/{}_mip.png".format(subject_id), proj_out)


def writeTxtLine(input_path, values):
    
    with open(input_path, "a") as f:
        f.write("{}".format(values[0]))

        for i in range(1, len(values)):
            f.write(",{}".format(values[i]))

        f.write("\n")


# Mean intensity projection
def formatMip(img, out, label_mask):

    # Project water signal intensities
    img_proj = normalize(np.sum(img, axis=1).astype("float"))

    # Prepare coloured overlay
    proj_out = np.zeros((img_proj.shape[0], img_proj.shape[1], 3))

    # Use label mask to identify components
    label_proj_left = normalize(np.sum(label_mask == 1, axis=1).astype("float"))
    label_proj_right = normalize(np.sum(label_mask == 2, axis=1).astype("float"))
    label_proj_scrap = normalize(np.sum(label_mask > 2, axis=1).astype("float"))

    # Blue: Left kidney, Green: Scrap volume, Red: Right kidney
    proj_out[:, :, 0] = 0.5 * img_proj + 0.5 * label_proj_left
    proj_out[:, :, 1] = 0.5 * img_proj + 0.5 * label_proj_scrap
    proj_out[:, :, 2] = 0.5 * img_proj + 0.5 * label_proj_right

    proj_out = (normalize(proj_out)*255).astype("uint8")
    proj_out = np.rot90(proj_out)

    return proj_out


# Copy the 3d segmentation by one voxel along longitudinal axis
# Form sum of absolute differences to rate smoothness
def rateSegmentationSmoothness(seg):

    seg_0 = np.zeros(seg.shape)

    # Get abs difference to vertically shifted copy 
    seg_0[:, :, 1:] = seg[:, :, :-1]
    dif_0 = np.sum(np.abs(seg - seg_0))

    size = np.sum(seg)

    if size == 0:
        rating = -1
    else:
        rating = -dif_0 / size

    return rating


def normalize(values):

    if len(np.unique(values)) > 1:
        values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
    else:
        values[:] = 0

    return values


def getDataloader(path_dicom, subject_ids, station_ids):

    dataset = DicomDataset(path_dicom, subject_ids, station_ids)

    loader = torch.utils.data.DataLoader(dataset,
                                        num_workers=8,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=True)
    return loader


class DicomDataset(data.Dataset):

    def __init__(self, path_dicom, subject_ids, station_ids):

        self.path_dicom = path_dicom
        self.subject_ids = subject_ids
        self.station_ids = station_ids

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):

        # Load intensity image
        subject_id = self.subject_ids[index]

        (station_vols, headers) = dicomToVolume.dicomToVolume(self.path_dicom + "{}_20201_2_0.zip".format(subject_id), self.station_ids)

        for i in range(len(station_vols)):
            station_vols[i] = np.ascontiguousarray(station_vols[i])

        return station_vols, headers, subject_id


if __name__ == '__main__':
    main(sys.argv)
