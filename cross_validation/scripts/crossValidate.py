import sys
import os
import time

import glob
import shutil

import numpy as np

import torch
import torch.utils.data as data

#
sys.path.insert(0, "models/")

from model_resUnet import UNet

import train
import predict
import evaluate
import dataLoading


def main(argv):

    path_network_out = "../networks/kidney_64_8fold/"

    path_training_slices = "../image_data/kidney_128/"
    path_split = "../splits/kidney_64_8fold/"

    # Paths to nrrd volumes
    path_stations_img = "/media/taro/DATA/Taro/UKBiobank/segmentations/kidney/combined_128/signals/NRRD/"
    path_stations_gt = "/media/taro/DATA/Taro/UKBiobank/segmentations/kidney/combined_128/segmentations/NRRD_fixedHeaders/"

    # Optional path to list of ids which are to be used as additional training samples on each split.
    # Set to None for conventional cross-validation
    path_train_ids_add = None

    runExperiment(path_network_out, path_training_slices, path_split, path_stations_img, path_stations_gt, path_train_ids_add)


def runExperiment(path_network_out, path_training_slices, path_split, path_stations_img, path_stations_gt, path_train_ids_add):

    I = 80000 # Training iterations
    save_step = 5000 # Iterations between checkpoint saving

    I_reduce_lr = 60000 # Reduce learning rate by factor 10 after this many iterations

    channel_count = 3 # Number of input channels
    class_count = 2 # Number of labels, including background
    class_weights = torch.FloatTensor([1, 1]) # Background, L1, L2...

    start_k = 0 # First cross-validation set to validate against

    do_train = True
    do_predict = True

    # Create folders
    if do_train and start_k == 0 and os.path.exists(path_network_out):
        print("ABORT: Network path already exists!")
        sys.exit()
        #shutil.rmtree(path_network_out)

    # Create folders and documentation when starting from scratch
    if do_train and start_k == 0:
        os.makedirs(path_network_out)
        createDocumentation(path_network_out, path_split)

    # Parse split
    split_files = [f for f in os.listdir(path_split) if os.path.isfile(os.path.join(path_split, f))]
    split_files = [f for f in split_files if "images_set" in f]

    K = len(split_files)
    cv_subsets = np.arange(K)

    #
    for k in range(start_k, K):

        # Validate against subset k
        val_subset = cv_subsets[k]

        # Train on all but subset k
        train_subsets = [x for f,x in enumerate(cv_subsets) if f != k]

        print("########## Validating against subset {}".format(val_subset))

        #
        path_out_k = path_network_out + "subset_{}/".format(val_subset)
        path_checkpoints = path_out_k + "snapshots/"

        if do_train or do_predict:
            print("Initializing network...")
            net = UNet(channel_count, class_count).cuda()

        if do_train:

            os.makedirs(path_out_k)
            os.makedirs(path_checkpoints)

            loader_train = getDataloader(path_training_slices + "data/", path_out_k + "train_files.txt", train_subsets, path_split, B=1, sigma=2, points=8, path_train_ids_add=path_train_ids_add)
            time = train.train(net, loader_train, I, path_checkpoints, save_step, class_weights, I_reduce_lr)

            with open(path_out_k + "training_time.txt", "w") as f: f.write("{}".format(time))

        if do_predict:
            evaluate.evaluateSnapshots(path_checkpoints, path_stations_img, path_stations_gt, path_split, val_subset, path_out_k + "eval/", net)

        evaluate.writeSubsetTrainingCurve(path_out_k)
        
    evaluate.aggregate(path_network_out, I, save_step)


# Copy scripts to network project folder as documentation
def createDocumentation(network_path, split_path):

    os.makedirs(network_path + "documentation")
    for file in glob.glob("*.py"): shutil.copy(file, network_path + "documentation/")

    os.makedirs(network_path + "split")
    for file in glob.glob(split_path + "*"): shutil.copy(file, network_path + "split/")


#
def getDataloader(input_path, output_path, subsets, path_split, B, sigma, points, path_train_ids_add):

    # Get chosen volumes
    subject_ids = []
    for k in subsets:
        subset_file = path_split + "images_set_{}.txt".format(k)

        with open(subset_file) as f: entries = f.readlines()
        subject_ids.extend(entries)

    # Add optional training images if specified
    if not path_train_ids_add is None:
        with open(path_split + path_train_ids_add) as f: entries = f.readlines()
        subject_ids.extend(entries)

    subject_ids = [f.replace("\n","") for f in subject_ids]    

    print("Loading data for {} subjects".format(len(subject_ids)))

    # For each subject, use stations 1 and 2
    stations = []
    stations.extend([f + "_station1" for f in subject_ids])
    stations.extend([f + "_station2" for f in subject_ids])

    # Get training samples
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    files = [f for f in files if f.split("_slice")[0] in stations]

    paths_seg = [input_path + f for f in files if "seg.npy" in f]
    paths_img = [f.replace("seg.npy", "img.npy") for f in paths_seg]

    print("Found {} samples...".format(len(paths_img)))

    #
    dataset = dataLoading.SliceDatasetDeformable(paths_img, paths_seg, sigma, points)

    loader = torch.utils.data.DataLoader(dataset,
                                        num_workers=8,
                                        batch_size=B,
                                        shuffle=True,
                                        pin_memory=True)

    # Document actually used training files
    with open(output_path, "w") as f:
        for img_file in paths_img:
            f.write("{}\n".format(img_file))
            f.write("{}\n".format(img_file.replace("img.npy", "seg.npy")))

    return loader


if __name__ == '__main__':
    main(sys.argv)
