import numpy as np
import nibabel as nib
import h5py
import os
from PIL import Image


img = nib.load(r"\raw_data\img\0066\DET0000101_avg.nii.gz")
label = nib.load(r"\raw_data\label\0066\DET0000101_avg_seg.nii.gz")

#Convert them to numpy format,
data = img.get_fdata()
label_data = label.get_fdata()

"""
Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], 
and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
"""
#clip the images within [-125, 275],
data_clipped = np.clip(data, -125, 275)

#normalize each 3D image to [0, 1], and
data_normalised = (data_clipped - (-125)) / (275 - (-125))

#extract 2D slices from 3D volume for training cases while
# e.g. slice 000
for i in range(data_clipped.shape[2]):
    formattedi = "{:03d}".format(i)
    slice000 = data_normalised[:,:,i]*255
    np.savetxt(r"label.txt", label_data[:, :, 6], delimiter=',', fmt='%5s')
    label_slice000 = label_data[:, :, i] * 255

    print(slice000.shape, type(slice000))

    image = Image.fromarray(slice000)
    image = image.convert("L")

    label = Image.fromarray(label_slice000)
    label = label.convert("L")

    image.save("./chestminist/" +"DET0000101_avg"+ str(i) + ".png")
    label.save("./chestminist/" + "DET0000101_avg" + str(i) + "_label.png")
