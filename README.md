# Neural networks for semantic segmentation of UK Biobank neck-to-knee body MRI

This repository contains PyTorch code for cross-validation and inference with neural networks for kidney segmentation on UK Biobank neck-to-knee body MRI, as described in:
[_"Kidney segmentation in neck-to-knee body MRI of 40,000 UK Biobank participants"_](https://arxiv.org/abs/2006.06996) [1]

The included inference pipeline and trained snapshot enables measurements of left and right parenchymal kidney volumes (excluding cysts and vessels) from these images.

Contents:
- 2.5D U-Net architecture with residual connections
- Infrastructure for training and *cross-validation*
- Pipeline for *inference* on neck-to-knee body MRI DICOMs
- Code for *quality_controls* based on numerical metrics
- Trained snapshot for parenchymal kidney tissue can be found at *TODO*

For any questions and suggestions, contact me at: taro.langner@surgsci.uu.se

# Citation
If you use this code for any derived work, please consider citing [1] and linking this GitHub.

# References

[1] [_T. Langner, A. Östling, L. Maldonis, A. Karlsson, D. Olmo, D. Lindgren, A. Wallin, L. Lundin, R. Strand, H. Ahlström, J. Kullberg, “Identifying morphological indicators of aging with neural networks on large-scale whole-body MRI,” arXiv:2006.06996 [cs], Jun. 2020. arXiv: 2006.06996_](https://arxiv.org/abs/2006.06996)\
