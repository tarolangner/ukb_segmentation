# Inference
The inference pipeline applies a trained network to DICOM files of the UK Biobank neck-to-knee body MRI (field 20201) to measure left and right parenchymal kidney volume.

The pipeline performs the following steps iteratively, it:
-Opens the neck-to-knee body MRI DICOM 
-Loads the water signal for the second and third imaging stations
-Applies the network for slice-wise predictions 
-Fuses the resulting binary station volumes
-Appends kidney measurements and quality metrics to a text file
-Optionally: Writes output volumes and mean intensity projections

# Notes
Note that the quality controls are applied only afterwards.
The runtimes reported in the paper were achieved by storing the signal information contained in the DICOM zips on a USB-SSD drive.
