# Quality Controls
The inference pipeline generates numerical quality ratings, which are used here to exclude potential outliers.
The quality control excludes those cases with presumed worst image quality (stage 1) and worst segmentation quality (stage 2).

First, run *defineFilters.py* on a previously generated inference folder. Next, run *applyExclusions.py* to mask out those ids flagged for exclusion.

# Algorithmic Quality Ratings
The inference pipeline generates two text files: *measurements.txt* and *quality.txt*. From these files, the numerical ratings for each subject are extracted:

## Image quality
-Image fusion cost
-Segmentation fusion cost
-Location cost

## Segmentation quality
-Segmentation smoothness
-Scrap volume
