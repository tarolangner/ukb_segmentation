import sys
import os

import numpy as np

import time

import shutil
import glob

import matplotlib.pyplot as plt
import nrrd

from sklearn.metrics import r2_score
import torch

sys.path.insert(0, "../../image_fusion/")
sys.path.insert(0, "models/")

import dicomToVolume
import fuseVolumes
import predictForSubject


# Evaluate mean cross-validation performance over all K subsets
def aggregate(input_path, I, save_step):

    if not os.path.exists(input_path + "eval/"):
        os.makedirs(input_path + "eval/")
        os.makedirs(input_path + "eval/volumes/")

    # Get subset folders
    subsets = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    subsets = [f for f in subsets if "subset_" in f]

    S = int(I // save_step) # Number of checkpoints
    K = len(subsets) # Number of subsets

    # Prepare mean metrics per checkpoint
    checkpoint_it = np.zeros(S)
    checkpoint_dice = np.zeros(S)
    checkpoint_mae = np.zeros(S)
    checkpoint_smape = np.zeros(S)
    checkpoint_r2 = np.zeros(S)
    checkpoint_loa_low = np.zeros(S)
    checkpoint_loa_high = np.zeros(S)

    # For each checkpoint
    for s in range(S):

        iteration = "00000{}".format(save_step * (s+1))[-6:]

        checkpoint_it[s] = iteration

        # Get subject-wise eval for given checkpoint from all subsets
        (ids, dices, p, tp, fp, voxel_scales) = aggregateEvalForIteration(input_path, subsets, iteration)

        # Write aggregated subject-wise values to main directory
        path_eval_s = input_path + "eval/eval_{}.txt".format(iteration)
        with open(path_eval_s, "w") as f:

            f.write("eid,dice,p,tp,fp,voxel_dim_cm3\n")

            for i in range(len(ids)):
                f.write("{},{},{},{},{},{}\n".format(ids[i], dices[i], p[i], tp[i], fp[i], voxel_scales[i]))

        # Calculate ground truth and predicted volumes
        volumes_gt = np.multiply(p, voxel_scales)
        volumes_out = np.multiply(tp + fp, voxel_scales)

        # Write subject-wise volumes to main eval directory
        path_vol_s = input_path + "eval/volumes_{}.txt".format(iteration)
        with open(path_vol_s, "w") as f:

            f.write("eid,gt_in_cm3,out_in_cm3\n")

            for i in range(len(ids)):
                f.write("{},{},{}\n".format(ids[i], volumes_gt[i], volumes_out[i]))

        ### Calculate mean performance metrics for checkpoint
        # Dice
        checkpoint_dice[s] = np.mean(dices)

        # MAE
        dif = volumes_gt - volumes_out
        checkpoint_mae[s] = np.mean(np.abs(dif))

        # SMAPE
        means = np.mean(np.vstack((volumes_gt, volumes_out)), axis=0)
        checkpoint_smape[s] = np.mean(np.abs(dif) / means) * 100

        # R^2
        checkpoint_r2[s] = r2_score(volumes_gt, volumes_out)

        # LoA
        dif_mean = np.mean(dif)
        dif_stdev = np.std(dif, ddof=1)

        checkpoint_loa_low[s] = dif_mean - 1.96 * dif_stdev
        checkpoint_loa_high[s] = dif_mean + 1.96 * dif_stdev

    #
    plotCurve(input_path + "dice_curve.png", checkpoint_it, checkpoint_dice, "Iterations", "Mean Dice")

    # Write metric files
    with open(input_path + "mean_performance_by_iteration.txt", "w") as f:
        f.write("iteration,mean_dice,mae,smape,r2,loa_low,loa_high\n")
        for i in range(len(checkpoint_it)):
            f.write("{},{},{},{},{},{},{}\n".format(int(checkpoint_it[i]), checkpoint_dice[i], checkpoint_mae[i], checkpoint_smape[i], checkpoint_r2[i], checkpoint_loa_low[i], checkpoint_loa_high[i]))

    # Move volumes and final predictions to main eval
    for k in range(K):
        path_volumes_k = input_path + subsets[k] + "/eval/volumes/"

        for file in glob.glob(path_volumes_k + "*"):
            shutil.move(file, input_path + "eval/volumes/" + os.path.basename(file))


def aggregateEvalForIteration(input_path, subset_folders, iteration):

    # Subject-wise evaluation results for given checkpoint
    ids = []
    dices = []
    p = []
    tp = []
    fp = []
    voxel_scales = []

    K = len(subset_folders)

    # For given checkpoint, aggregate evaluation metrics from all subsets
    for k in range(K):
        
        path_eval_s_k = input_path + subset_folders[k] + "/eval/eval_{}.txt".format(iteration)

        (ids_k, dices_k, p_k, tp_k, fp_k, voxel_scales_k) = parseEvalFile(path_eval_s_k)

        ids.extend(ids_k)
        dices.extend(dices_k)
        p.extend(p_k)
        tp.extend(tp_k)
        fp.extend(fp_k)
        voxel_scales.extend(voxel_scales_k)

    ids = np.array(ids).astype("int")
    dices = np.array(dices).astype("float")
    p = np.array(p).astype("float")
    tp = np.array(tp).astype("float")
    fp = np.array(fp).astype("float")
    voxel_scales = np.array(voxel_scales).astype("float")

    return (ids, dices, p, tp, fp, voxel_scales)


def parseEvalFile(input_path):

    # Read evaluation file
    with open(input_path) as f: entries = f.readlines()
    entries.pop(0)

    # Extract columns
    ids = np.array([f.split(",")[0] for f in entries]).astype("int")
    values_dice = np.array([f.split(",")[1] for f in entries]).astype("float")

    values_p = np.array([f.split(",")[2] for f in entries]).astype("float")
    values_tp = np.array([f.split(",")[3] for f in entries]).astype("float")
    values_fp = np.array([f.split(",")[4] for f in entries]).astype("float")

    values_voxel_dim = np.array([f.split(",")[5] for f in entries]).astype("float")

    return (ids, values_dice, values_p, values_tp, values_fp, values_voxel_dim)


def writeSubsetTrainingCurve(path_out):

    path_eval = path_out + "eval/"

    files = [f for f in os.listdir(path_eval) if os.path.isfile(os.path.join(path_eval, f))]
    files = [f for f in files if "eval_" in f and ".txt" in f]

    S = len(files)

    values_it = np.zeros(S)
    values_dice = np.zeros(S)

    # For all checkpoints
    for i in range(S):

        # Get evaluation file and extract iteration
        path_eval_s = path_eval + files[i]
        values_it[i] = float(files[i].split("_")[1].split(".")[0])

        # Read evaluation file
        with open(path_eval_s) as f: entries = f.readlines()
        entries.pop(0)

        # Extract dice and get mean 
        dices = [f.split(",")[1] for f in entries]
        values_dice[i] = np.mean(np.array(dices).astype("float"))

    plotCurve(path_out + "dice_curve.png", values_it, values_dice, "Iterations", "Mean Dice")


def plotCurve(path_out, x, y, label_x, label_y):

    x = np.array(x).astype("int")
    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]

    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.savefig(path_out, dpi=150)
    plt.close()


def evaluateSnapshots(path_checkpoints, path_stations_img, path_stations_gt, path_split, val_subset, path_out, net, station_ids, target_spacing):

    time_start = time.time()

    # Get subjects ids of validation set
    subset_file = path_split + "images_set_{}.txt".format(val_subset)

    with open(subset_file) as f: entries = f.readlines()
    val_subjects = np.array(entries).astype("int")

    N = len(val_subjects)

    #
    if not os.path.exists(path_out): 
        os.makedirs(path_out)
        os.makedirs(path_out + "volumes/")

    # Fuse and store reference segmentation
    for i in range(N):
        fuseStationsGt(val_subjects[i], path_stations_img, path_stations_gt, path_out + "volumes/", station_ids, target_spacing)

    # Find checkpoints
    checkpoint_files = [f for f in os.listdir(path_checkpoints) if os.path.isfile(os.path.join(path_checkpoints, f))]
    checkpoint_files = [f for f in checkpoint_files if ".pth.tar" in f]

    S = len(checkpoint_files)

    for i in range(S):

        print("   Evaluating snapshot {}...".format(i))
        checkpoint_i = path_checkpoints + checkpoint_files[i]

        predictWithCheckpoint(checkpoint_i, path_stations_img, val_subjects, net, path_out + "volumes/", station_ids, target_spacing)

        iteration = checkpoint_files[i].split("_")[1].split(".")[0]
        evaluateAgreement(path_out, iteration, val_subjects)
        
    time_end = time.time()
    print("    Time elapsed: {}".format(time_end - time_start))


def evaluateAgreement(path_out, iteration, val_subjects):

    N = len(val_subjects)

    with open(path_out + "eval_{}.txt".format(iteration), "w") as f:

        f.write("eid,dice,p,tp,fp,voxel_dim_cm3\n")
        
        for i in range(N):

            (gt, header) = nrrd.read(path_out + "volumes/{}_gt.nrrd".format(val_subjects[i]))
            (out, header) = nrrd.read(path_out + "volumes/{}_out.nrrd".format(val_subjects[i]))

            # Ensure that reference is binary
            gt = np.around(gt)
            
            ###
            # Get physical voxel dimensions
            space_dir = header["space directions"]
            voxel_dim = np.array((space_dir[0][0], space_dir[1][1], space_dir[2][2]))
            voxel_scale = np.prod(voxel_dim) / (10*10*10)


            # Get positives, true positives, false positives
            p = np.count_nonzero(gt)
            tp = np.count_nonzero(np.multiply(gt, out))
            fp =  np.count_nonzero(np.multiply(1 - gt, out))

            #
            dice = 1.0
            divisor = float(np.sum(p) + np.sum(tp) + np.sum(fp))
            if divisor != 0:
                dice = 2 * np.sum(tp) / divisor

            f.write("{},{},{},{},{},{}\n".format(val_subjects[i],dice,p,tp,fp,voxel_scale))


def predictWithCheckpoint(path_checkpoint, path_stations_img, val_subjects, net, path_out, station_ids, target_spacing):

    # Load network weights
    checkpoint = torch.load(path_checkpoint, map_location={"cuda" : "cpu"})
    net.load_state_dict(checkpoint['state_dict'])

    #
    N = len(val_subjects)

    #
    for i in range(N):

        print("Subject {}".format(val_subjects[i]))

        stations = []
        headers = []

        for s in station_ids:

            (station, header) = nrrd.read(path_stations_img + "{}_station{}_W.nrrd".format(val_subjects[i], s))
            stations.append(station)
            headers.append(header)
            #(img_2, header_2) = nrrd.read(path_stations_img + "{}_station2_W.nrrd".format(val_subjects[i]))

        if not os.path.exists(path_out + "{}_img.nrrd".format(val_subjects[i])):
            fuse_img = True
        else:
            fuse_img = False

        (img, out, header, _, _) = predictForSubject.predictForSubject(stations, headers, net, target_spacing, fuse_img)

        if fuse_img:
            nrrd.write(path_out + "{}_img.nrrd".format(val_subjects[i]), img, header, compression_level=1)

        nrrd.write(path_out + "{}_out.nrrd".format(val_subjects[i]), out, header, compression_level=1)


def fuseStationsGt(subject_id, path_stations_img, path_stations_gt, path_out, station_ids, target_spacing):

    volumes_gt = []
    headers_gt = []
    positions = []
    spacings = []

    for s in station_ids:

        path_s = path_stations_gt + "{}_station{}.nrrd".format(subject_id, s)

        if not os.path.exists(path_s): 
            print("WARNING: Could not find ground truth segmentation, assuming empty segmentation for {}".format(path_s))
            path_s = path_stations_img + "{}_station{}_W.nrrd".format(subject_id, s)

            # Load signal instead and set values to 0
            (volume_gt, header) = nrrd.read(path_s)
            volume_gt[:] = 0

        else:
            (volume_gt, header) = nrrd.read(path_s)

            # Round volumes to binarize segmentations from SmartPaint. Using the float values appears to provide no benefit
            volume_gt = np.around(volume_gt)

        volumes_gt.append(volume_gt)
        headers_gt.append(header)

        #
        positions.append(header["space origin"])

        spacing = header["space directions"]
        spacing = np.array((spacing[0][0], spacing[1][1], spacing[2][2]))

        spacings.append(spacing)

    #
    (gt, gt_origin,seg_fusion_cost) = fuseVolumes.fuseStations(volumes_gt, positions, spacings, target_spacing, False)

    header = headers_gt[0]
    header["sizes"] = gt.shape
    header["space origin"] = gt_origin
    for i in range(3): header["space directions"][i][i] = target_spacing[i]

    nrrd.write(path_out + "{}_gt.nrrd".format(subject_id), gt, header, compression_level=1)
