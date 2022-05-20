![image](https://user-images.githubusercontent.com/66905164/169608468-72869a6d-d95f-4dd1-944e-c7518d164482.png)


# ViTUResNet-Based-Medical-Image-Segmentation
ViTUResNet is a hybrid CNN-Transformer architecture with the backbone of TransUNet.
The experiments were conducted on Synapse multi-organ segmentation dataset.

- Access to the synapse multi-organ dataset:

  -  Sign up in the official https://www.synapse.org/ and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
  - It is possible to request the preprocessed dataset from the original repo authors.



![git](https://user-images.githubusercontent.com/66905164/169605613-5269fa05-eb9b-4936-b616-9068b75113b2.png)


- The architecure is given below for better understanding

![architecture](https://user-images.githubusercontent.com/66905164/169606363-e8910bf8-6427-418a-af04-c0c50c8f6c62.png)


Vision Transformer ( You can watch this repo https://github.com/RadeenXALNW/Vision-Transformer-ViT ) for better understanding the intuition 


![image](https://user-images.githubusercontent.com/66905164/169606905-d0de58a9-f66d-47c4-b741-7e53804f6689.png)

