import os
import sys

import elasticdeform

import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np

import scipy
from scipy import ndimage

class SliceDatasetDeformable(data.Dataset):

    ##
    # img_paths is list of paths to intensity images
    # seg_paths is list of paths to segmentation images, define as None if no segmentations exist
    # sigma is deformation intensity, points the number of coordinates for grid deformation
    def __init__(self, img_paths, seg_paths, sigma, points):

        self.seg_paths = seg_paths
        self.img_paths = img_paths
        self.sigma = sigma
        self.points = points

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        # Load intensity image
        img_path = self.img_paths[index]
        img = np.load(img_path)
        seg_exists = len(self.seg_paths) > 0

        # Use elastic deformation https://pypi.org/project/elasticdeform/

        # Load segmentation image if exists
        if seg_exists:
            seg_path = self.seg_paths[index]
            seg = np.load(seg_path)

            if self.sigma != 0:

                # Form stack of image and segmentation and deform
                def_input = np.concatenate((img, seg.reshape(1, seg.shape[0], seg.shape[1])), axis=0)
                def_output = elasticdeform.deform_random_grid(def_input, self.sigma, self.points, order=1, axis=(1,2))

                # Separate image and segmentation in deformed stack
                img = def_output[:img.shape[0], :, :]
                seg = np.around(def_output[img.shape[0], :, :])

        elif self.sigma != 0:
            img = elasticdeform.deform_random_grid(img, self.sigma, self.points, order=1, axis=(1,2))

        # Convert images to pytorch tensors
        X = Variable(torch.from_numpy(img)).float()
    
        if seg_exists:
            Y = Variable(torch.from_numpy(seg)).long()

        else:
            Y = torch.zeros(1) # dummy segmentation

        name = os.path.basename(self.img_paths[index])

        return X, Y, name

