# code
Body composition machine learning

Read the information from the image feature files, such as pixel size, voxel space, and mask label values.
Read the directory structure:

TEXT
Project  
├─0_Read_subfolders.py  
├──NRRD  
    │    ├──1  
    │    │    ├─6 Body 5.0 CE.nrrd  
    │    │    ├─Segmentation_1.seg.nrrd  
    │    ├──2  
    │    │    ├─3 5mm C.nrrd  
    │    │    ├─Segmentation_2.seg.nrrd  
    │    ├──3  
    │    │    ├─2 Unnamed Series.nrrd  
    │    │    ├─Segmentation.seg.nrrd  
Calculate the distribution of Hounsfield Unit (HU) values and plot a histogram for observation.

Perform image resampling and grayscale value correction.

Extract features.（Use python3.7.1 pyradiomics）

Calculate the consistency of feature variables and eliminate those with intra-group or inter-group consistency below 0.75.

Select appropriate feature variables to include in the model (using MRMR).

Adjust the hyperparameters of the model and screen for better-performing models.

Plot the variable importance graph of the optimal model and generate evaluation result images.
