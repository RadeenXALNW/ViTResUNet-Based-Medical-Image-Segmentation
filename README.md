# ViTUResNet-Based-Medical-Image-Segmentation
ViTUResNet is a hybrid CNN-Transformer architecture with the backbone of TransUNet.
The experiments were conducted on Synapse multi-organ segmentation dataset.

- Access to the synapse multi-organ dataset:

  -  Sign up in the official # https://www.synapse.org/ and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
  - It is possible to request the preprocessed dataset from the original repo authors.
  - Set up a Google Cloud Project to store your data in a bucket.
  - Convert the data from numpy to TfRecords (Tensorflow’s binarystorage format) to speed up training and enable parallel data reading from disk. We provide a data parsing pipeline to write and read TfRecords as a TFDataset in the module data_processing. A guide notebook is available under experiments/create_tfds_records.ipynb.
The directory structure of the whole project is as follows:

! [](D:/deep learning research paper/cu conference/git.png)
