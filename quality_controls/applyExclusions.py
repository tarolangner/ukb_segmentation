import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
from sklearn.metrics import r2_score

#####
# 

def main(argv):

    path_inference = "/media/taro/DATA/Taro/Projects/ukb_segmentation/github/temp_out/"

    # Get all excluded ids
    ids_out = aggregateExcludedIds(path_inference)

    # Mask quality and measurement text files
    _ = maskFile(path_inference, "quality.txt", ids_out)
    N = maskFile(path_inference, "measurements.txt", ids_out)

    N_out = len(ids_out)
    print("Excluded {} of {} ({}%)".format(N_out, N, 100*N_out / float(N)))

    # Write documentation
    with open(path_inference + "quality_controls/exclusion_counts.txt", "w") as f:
        f.write("In: {}\n".format(N))
        f.write("exclude: {} ({}%)\n".format(N_out, 100*N_out/ float(N)))
        f.write("Out: {}\n".format(N - N_out))


def maskFile(path_inference, file_name, ids_out):

    with open(path_inference + file_name) as f: entries = f.readlines()
    header = entries[0]

    body_out = [f for f in entries[1:] if int(f.split(",")[0]) not in ids_out]

    with open(path_inference + "/quality_controls/" + file_name, "w") as f:
        f.write(header)
        for i in range(len(body_out)):
            f.write(body_out[i])

    return len(entries) - 1 


def aggregateExcludedIds(path_inference):

    input_path = path_inference + "quality_controls/exclusions/"

    if not os.path.exists(input_path):
        print("ERROR: Found no id lists for exclusion!")
        sys.exit()

    files = [input_path + f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    ids_out = []

    for i in range(len(files)):

        path_i = files[i]

        with open(path_i) as f: entries = f.readlines()
        ids = [f.split(",")[0] for f in entries]

        ids_out.extend(ids)

    ids_out = np.array(ids_out).astype("int")
    ids_out = np.unique(ids_out)

    return ids_out


if __name__ == '__main__':
    main(sys.argv)
