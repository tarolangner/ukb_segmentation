# Cross-validation
Using the code samples in the *scripts* subfolder, a segmentation model can be trained and evaluated in several steps:

1. Run *createNewSplit.py* to generate a K-fold cross-validation split in the directory "splits"
2. Run *createTrainingSlices.py* to extract 2D input and ground truth samples in .npy format from 3D .nrrd files, stored in the directory "image_data"
3. Run *crossValidate.py* to train and evaluate a model, with the results stored to the directory "networks"

To re-train a network for inference using all data, the cross-validation split can simply be set to contain all images in one split set.

Note that the first run on new training data may be very slow, whereas subsequent runs will benefit from caching by the data loader.
