import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import shutil


#####
# Sort the numerical quality ratings and extract the worst cases.
# Store the quality rating curves, the worst IDs and mean intensity projections.
# First stage rates image quality, second stage rates segmentation quality.

def main(argv):

    # Path to completed inference run
    path_inference = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/temp_out/"

    # Output paths for excluded ids, plots and mean intensity projections
    path_out = path_inference + "quality_controls/"
    path_out_exc = path_out + "exclusions/" 
    path_out_mip = path_out + "MIP_excluded/"

    # Create output folders
    if not os.path.exists(path_out): os.makedirs(path_out)
    if not os.path.exists(path_out_exc): os.makedirs(path_out_exc)
    if not os.path.exists(path_out_mip): os.makedirs(path_out_mip)

    # Read ratings from quality.txt and measurements.txt
    (ids, img_fuse_cost, seg_fuse_cost, seg_smoothness, location_cost, scrap_volume) = parseRatings(path_inference)
    N = len(ids)

    ###
    # Stage 1: Rate image quality (first float is share of retained cases)
    ids_out_img_fuse_cost = rejectPercentile(0.99, ids, img_fuse_cost, "img_fusion_cost", path_inference, path_out)
    ids_out_seg_fuse_cost = rejectPercentile(0.98, ids, seg_fuse_cost, "seg_fusion_cost", path_inference, path_out)
    ids_out_location_cost = rejectPercentile(0.99, ids, location_cost, "location_cost", path_inference, path_out)

    # Remove those ids flagged by first stage
    ids_out_stage1 = np.unique(np.hstack((ids_out_img_fuse_cost, ids_out_seg_fuse_cost, ids_out_location_cost)))
    print("Removing {0} unique ids in first stage ({1:0.3f}%)".format(len(ids_out_stage1), len(ids_out_stage1) / len(ids) * 100))

    mask = np.invert(np.in1d(ids, ids_out_stage1))
    ids = ids[mask]
    img_fuse_cost = img_fuse_cost[mask]
    seg_fuse_cost = seg_fuse_cost[mask]
    seg_smoothness = seg_smoothness[mask]
    location_cost = location_cost[mask]
    scrap_volume = scrap_volume[mask]

    ###
    # Stage 2: Rate segmentation quality (first float is share of retained cases)
    ids_out_seg_smoothness = rejectPercentile(0.99, ids, -seg_smoothness, "seg_smoothness", path_inference, path_out)
    ids_out_scrap_volume = rejectPercentile(0.99, ids, scrap_volume, "scrap_volume", path_inference, path_out)

    ids_out_stage2 = np.unique(np.hstack((ids_out_seg_smoothness, ids_out_scrap_volume)))
    print("Removing {0} unique ids in second stage ({1:0.3f}%)".format(len(ids_out_stage2), len(ids_out_stage2) / len(ids) * 100))

    N_out = len(ids_out_stage1) + len(ids_out_stage2)
    print("Total removals: {0} ({1:0.3f}%)".format(N_out, N_out / float(N) * 100))


def parseRatings(path_inference):

    (ids_quality, img_fuse_cost, seg_fuse_cost, seg_smoothness, location_cost) = parseQualityFile(path_inference + "quality.txt")
    (ids_measure, volume_total, volume_left, volume_right, dist) = parseMeasurements(path_inference + "measurements.txt")

    if not np.array_equal(ids_quality, ids_measure):
        print("ERROR: IDs in quality.txt and measurements.txt do not match!")
        sys.exit()

    scrap_volume = volume_total - volume_left - volume_right

    return (ids_quality, img_fuse_cost, seg_fuse_cost, seg_smoothness, location_cost, scrap_volume)


def parseQualityFile(input_path):
    
    with open(input_path) as f: entries = f.readlines()
    entries.pop(0)

    ids = [f.split(",")[0].replace("\"", "") for f in entries]
    img_fuse_cost = [f.split(",")[1].replace("\"", "") for f in entries]
    seg_fuse_cost = [f.split(",")[2].replace("\"", "") for f in entries]
    seg_smoothness = [f.split(",")[3].replace("\"", "").replace("\n", "") for f in entries]
    location_cost = [f.split(",")[4].replace("\"", "").replace("\n", "") for f in entries]

    ids = np.array(ids).astype("int")
    img_fuse_cost = np.array(img_fuse_cost).astype("float")
    seg_fuse_cost = np.array(seg_fuse_cost).astype("float")
    seg_smoothness = np.array(seg_smoothness).astype("float")
    location_cost = np.array(location_cost).astype("float")

    idx = np.argsort(ids)

    #
    ids = ids[idx]
    img_fuse_cost = img_fuse_cost[idx]
    seg_fuse_cost = seg_fuse_cost[idx]
    seg_smoothness = seg_smoothness[idx]
    location_cost = location_cost[idx]

    return (ids, img_fuse_cost, seg_fuse_cost, seg_smoothness, location_cost)
    

def parseMeasurements(input_path):

    with open(input_path) as f: entries = f.readlines()
    entries.pop(0)

    ids = [f.split(",")[0].replace("\"", "") for f in entries]
    volume_total = [f.split(",")[1].replace("\"", "") for f in entries]
    volume_left = [f.split(",")[2].replace("\"", "") for f in entries]
    volume_right = [f.split(",")[3].replace("\"", "") for f in entries]

    offsets_x = [f.split(",")[4].replace("\"", "") for f in entries]
    offsets_y = [f.split(",")[5].replace("\"", "") for f in entries]
    offsets_z = [f.split(",")[6].replace("\"", "").replace("\n", "") for f in entries]

    #
    ids = np.array(ids).astype("int")
    volume_total = np.array(volume_total).astype("float")
    volume_left = np.array(volume_left).astype("float")
    volume_right = np.array(volume_right).astype("float")

    offsets_x = np.array(offsets_x).astype("float")
    offsets_y = np.array(offsets_y).astype("float")
    offsets_z = np.array(offsets_z).astype("float")

    #
    offsets = np.vstack((offsets_x, offsets_y, offsets_z))
    dist = np.sqrt(np.sum(offsets * offsets, axis=0))

    #
    idx = np.argsort(ids)

    ids = ids[idx]
    volume_total = volume_total[idx]
    volume_left = volume_left[idx]
    volume_right = volume_right[idx]

    dist = dist[idx]

    return (ids, volume_total, volume_left, volume_right, dist)


def rejectPercentile(factor, ids, values, tag, path_inference, path_out):

    N = len(values)

    # Isolate top (1 - factor) percentile
    N_99 = int(factor*N)
    N_1 = N - N_99

    idx = np.argsort(values)
    ids_out = ids[idx[N_99:]]

    # Plot curve
    plt.title("{0}, cutoff: {1:0.3f}, excluded N={2} ".format(tag, values[idx[N_99]], N_1))
    plt.plot(np.arange(N_99+1), values[idx[:N_99+1]], color="C0")
    plt.plot(np.arange(N_1)+N_99, values[idx[N_99:]], color="C1")

    plt.savefig(path_out + "{}_exclusions.png".format(tag))
    plt.show(block=True)

    # Write excluded ids
    with open(path_out + "exclusions/excluded_{}.txt".format(tag), "w") as f: 
        for i in range(len(ids_out)):
            f.write("{}\n".format(ids_out[i]))

    # Copy excluded mean intensity projections
    if os.path.exists(path_out + "MIP_excluded/" + tag): 
        shutil.rmtree(path_out + "MIP_excluded/" + tag)
    
    os.makedirs(path_out + "MIP_excluded/" + tag)

    #
    print("Copying mean intensity projections...")
    for i in range(N_1):
        index = "0000{}".format(i)[-5:]
        id_i = ids[idx[-(i+1)]]
        mip_name = "{}_{}_mip.png".format(index, id_i)
        shutil.copyfile(path_inference + "MIP/{}_mip.png".format(id_i), path_out + "MIP_excluded/" + tag + "/" + mip_name)
    
    return ids_out


if __name__ == '__main__':
    main(sys.argv)
