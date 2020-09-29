import sys
import os
import time

import numpy as np

import torch
import torch.utils.data as data

#from model_unet import UNet
import train
import dataLoading


def predict(net, loader, class_count, output_path, device_id, write_overlay):

    net.eval()
    torch.backends.cudnn.benchmark = True

    for X, Y, slice_name in loader:

        X = X.cuda(device_id, non_blocking=True)
        output = net(X)

        # Get effective batch size
        B = X.size(0)

        img_B = X.cpu().data.numpy()
        gt_B = Y.cpu().data.numpy()
        output_B = output.cpu().data.numpy()

        # For all samples in batch
        for b in range(B):

            img = img_B[b, :, :, :]
            gt = gt_B[b, :, :]
            out = np.argmax(output_B[b, :, :, :], 0)

            if write_overlay:
                writeOverlay(img, gt, out, output_path) 


def writeOverlay(img, gt, out, output_path):

    # Pick central signal slice 
    img_w = img[1, :, :]

    # Color code labels
    tp = np.multiply(gt, out)
    fn = np.multiply(gt, 1 - out)
    fp = np.multiply(1 - gt, out)

    # Blue: True positive
    # Red: False negative
    # Yellow: False positive
    overlay = np.zeros((out.shape[0], out.shape[1], 3))
    overlay[:, :, 0] = img_w + tp # blue
    overlay[:, :, 1] = img_w + fp # green
    overlay[:, :, 2] = img_w + fn + fp # red

    #
    overlay = (normalize(overlay) * 255).astype("uint8")
    cv2.imwrite(output_path + "overlay/" + slice_name[b].replace("img.npy", "out.png"), overlay)

    # Signal only
    if False:
        # (c, y, x) to (x, y, c)
        img_out = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
        img_out = (normalize(img_out) * 255).astype("uint8")

        cv2.imwrite(output_path + "overlay/" + slice_name[0].replace("img.npy", "img.png"), img_out)


# Safely normalize to [0, 1]
def normalize(values):

    if len(np.unique(values)) > 1:
        values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
    else:
        values[:] = 0

    return values


def getDataloader(input_path, class_count):

    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    paths_img = [input_path + f for f in files if "img.npy" in f and "aug" not in f]
    print("Found {} images...".format(len(paths_img)))

    dataset = dataLoading.SliceDataset(paths_img, [], np.zeros(3), class_count)

    loader = torch.utils.data.DataLoader(dataset,
                                        num_workers=8,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=True)

    return loader
