# ViTUResNet-Based-Medical-Image-Segmentation
ViTUResNet is a hybrid CNN-Transformer architecture with the backbone of TransUNet.
The experiments were conducted on Synapse multi-organ segmentation dataset.

- Access to the synapse multi-organ dataset:

 -  Sign up in the official Synapse website and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
It is possible to request the preprocessed dataset from the original repo authors.
 -Set up a Google Cloud Project to store your data in a bucket.
Convert the data from numpy to TfRecords (Tensorflow’s binarystorage format) to speed up training and enable parallel data reading from disk. We provide a data parsing pipeline to write and read TfRecords as a TFDataset in the module data_processing. A guide notebook is available under experiments/create_tfds_records.ipynb.
The directory structure of the whole project is as follows:

├───data
│   ├───synapse-train-224
│   		├── record_0.tfrecords
│   		└── *.tfrecords
|   ├───synapse-test-224
│   		├── case0001.tfrecords
│   		└── *.tfrecords
│   ├───test_vol_h5
│   		├── case0001.npy.h5
│   		└── *.npy.h5
│   └───train_npz
│   		├── case0005_slice000.npz
│   		└── *.npz
├── TransUNet
    ├───data_processing
    │   ├───dataset_synapse.py
    │   ├───data_parser.py
    │   └───__init__.py
    ├───experiments
    │	├───config.py
    │   ├───create_tfds_records.ipynb
    │   ├───data_exploration.ipynb
    │   └───__init__.py
    ├───models
    │   ├───decoder_layers.py
    │   ├───encoder_layers.py
    │   ├───resnet_v2.py
    │   ├───transunet.py
    │   ├───utils.py
    │   └───__init__.py
    ├───synapse_ct_scans
    │   ├───case0022.tfrecords
    │   ├───case0025.tfrecords
    │   └───case0029.tfrecords
    └───utils
        ├───evaluation.py
        ├───visualize.py
        └───__init__.py
