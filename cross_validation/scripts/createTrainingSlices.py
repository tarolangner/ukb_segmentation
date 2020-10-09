import os
import sys

import shutil
import numpy as np

import time

import nrrd

# Target dimensions for cropping/padding
c_x = 224
c_y = 192

# Skip certain axial slices at top and bottom due to folding artefacts
c_crop_slices = 3

###### 
# Extract training samples from .nrrd volumes and store them to .npy.
# For each axial slice:
#   Three adjacent axial slices are concatenated along the third dimension as 2.5D input sample.
#   One binary segmentation of the central slice serves as ground truth.

def main(argv):

    #
    #path_img = "/media/taro/DATA/Taro/UKBiobank/segmentations/kidney/combined_128/signals/NRRD/"
    #path_seg = "/media/taro/DATA/Taro/UKBiobank/segmentations/kidney/combined_128/segmentations/NRRD/"
    #path_ids = "/media/taro/DATA/Taro/UKBiobank/segmentations/kidney/combined_128/subject_ids.txt"
    #output_path = "../image_data/kidney_128/"

    #
    #path_img = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/temp_volumes/liver/signals/NRRD_3/"
    #path_seg = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/temp_volumes/liver/segmentations/NRRD_fixedHeaders/"
    #path_ids = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/temp_volumes/liver/ids.txt"
    #output_path = "../image_data/liver_allStations/"

    #
    path_img = "/media/taro/DATA/Taro/UKBiobank/segmentations/liver/Andres_refined/signals/"
    path_seg = "/media/taro/DATA/Taro/UKBiobank/segmentations/liver/Andres_refined/segmentations/"
    path_ids = "/media/taro/DATA/Taro/UKBiobank/segmentations/liver/Andres_refined/ids_add.txt"
    output_path = "../image_data/liver_refined_99_add/"

    #####
    createFolders(output_path, overwrite=True)

    # Copy this script and ids as documentation
    storeDocumentation(path_ids, output_path + "documentation/")

    # Read ids
    with open(path_ids) as f: ids = f.readlines()
    ids = np.array(ids).astype("int")

    # Get image volumes
    files_img = [f for f in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, f))]
    files_img = [f for f in files_img if "" in f]

    # Get ground truth segmentation volumes
    if not path_seg is None:
        files_seg = [f for f in os.listdir(path_seg) if os.path.isfile(os.path.join(path_seg, f))]
        files_seg = [f for f in files_seg if "" in f]

    #
    time_start = time.time()
    
    for i in range(len(ids)):
        convertSubject(ids[i], files_img, files_seg, path_img, path_seg, output_path)

    time_end = time.time()

    print("Elapsed time: {}".format(time_end - time_start))


# For each axial slice of this subject, extract an input sample and ground truth segmentation
def convertSubject(subject_id, files_img, files_seg, path_img, path_seg, output_path):
        
    # Get water signal image station volumes for given subject
    files_w = [f for f in files_img if str(subject_id) in f and "W.nrrd" in f]

    # For each station
    for j in range(len(files_w)):

        path_w = path_img + files_w[j]
        name = files_w[j].replace("_W.nrrd","")

        print(name)

        # Extract slice stacks and shape
        (slices_img, shape_img) = formatSignal(path_w)

        # Extract ground truth segmentation slices if desired
        if not path_seg is None:

            # Remove water signal name tag
            station_name = files_w[j].replace(".nrrd","").replace("_W", "")
            file_s = [f for f in files_seg if station_name in f]

            if not file_s:
                file_s = "nope"
            else:
                file_s = file_s[0]

            # Extract ground truth segmentation slices
            (slices_seg, shape_seg) = formatSeg(path_seg + file_s, shape_img)

            if not np.array_equal(shape_img, shape_seg):
                print("ERROR: Mismatching dimensions for img and seg of {} ({} vs {})".format(name, shape_img, shape_seg))
                sys.exit()

        # For each axial slice, save outputs
        for z in range(len(slices_img)):

            index = "000{}".format(z)[-4:]
            np.save(output_path + "data/" + name + "_slice_{}_img.npy".format(index), slices_img[z])
            
            if not path_seg is None:
                np.save(output_path + "data/" + name + "_slice_{}_seg.npy".format(index), slices_seg[z])


# For each axial slice of the given station volume, extract a 2.5D input sample
def formatSignal(path_w):

    (img_w, _) = nrrd.read(path_w)

    slices_out = []

    Z = img_w.shape[2]

    # For each axial slice
    for z in range(Z):

        # Skip some top and bottom slices due to folding artefacts
        if z < c_crop_slices or (img_w.shape[2] - z) <= c_crop_slices: continue

        # Get coordinates of three adjacent slices, periodic border condition
        z_range = np.clip(np.arange(-1,2)+z, c_crop_slices, Z-c_crop_slices-1)

        # Initialize sample, three adjacent axial slices
        slice_img = np.zeros((c_x, c_y, 3))

        # Extract slices with pre-processing
        for i in range(3):
            z_r = z_range[i]

            slice_w = img_w[:, :, z_r]
            slice_w = normalizeClip(slice_w)
            slice_w = padSlice(slice_w)

            slice_img[:, :, i] = slice_w

        # Change indexing order:
        # (x, y, c) to (c, y, x)
        slice_img = np.swapaxes(slice_img, 0, 2)
        slice_img = np.reshape(slice_img, (3, c_y, c_x))

        slices_out.append(slice_img)


    return (slices_out, img_w.shape)


# For each axial slice of the given station volume, extract a 2d ground truth segmentation slice
def formatSeg(path_seg, shape_img):

    if os.path.exists(path_seg):
        (seg, _) = nrrd.read(path_seg)
    else:
        print("No segmentation found, assuming all labels are background")
        seg = np.zeros(shape_img)

    slices_out = []

    Z = seg.shape[2]

    # For each axial slice
    for z in range(Z):

        # Skip some top and bottom slices due to folding artefacts
        if z < c_crop_slices or (seg.shape[2] - z) <= c_crop_slices: continue

        slice_seg = seg[:, :, z]

        # Round float values such as used by SmartPaint
        slice_seg = (np.around(slice_seg)).astype("float")
        slice_seg = padSlice(slice_seg)

        # Change indexing order:
        # (x, y) to (y, x)
        slice_seg = np.swapaxes(slice_seg, 0, 1)
        slices_out.append(slice_seg)


    return (slices_out, seg.shape)


def createFolders(output_path, overwrite):

    if os.path.exists(output_path):

        if overwrite:
            shutil.rmtree(output_path)
        else:
            print("ABORT: Output folder already exists!")
            sys.exit()

    os.makedirs(output_path)
    os.makedirs(output_path + "documentation/")

    os.makedirs(output_path + "data/")


def storeDocumentation(path_ids, output_path):

    shutil.copyfile(path_ids, output_path + "chosen_ids.txt")

    shutil.copyfile("createTrainingSlices.py", output_path + "createTrainingSlices.py")


# Normalize image intensities after clipping the brightest one percent for stability
def normalizeClip(values):
    
    values = values.astype("float")
    t = np.ceil(np.percentile(values, 99))

    values[values > t] = t
    
    if len(np.unique(values)) > 1:
        values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
    else:
        values[:] = 0

    return values
 

# Zero-pad image to (c_x)x(c_y)
def padSlice(values):

    target_shape = np.array((c_x, c_y))
    pad = ((target_shape - values.shape) / 2).astype("int")

    values = np.pad(values, ((pad[0], pad[0]), (pad[1], pad[1])), mode="constant", constant_values = 0)

    return values


if __name__ == '__main__':
    main(sys.argv)
