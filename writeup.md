# Semantic Segmentation
The objective of this project is to find the free space on road using a deep learning technique called Fully Convolutional Networks (link:). The dataset used for training of deep neural network is KITTI dataset (link:) which has two classes i.e. road area and non-road area. 

## Fully Convolutinal Networks
The CNNs are good for classification but they los the spatial information which means that the CNNs can tell if there is specific object present in image or not (when trained to classify the object) but can't tell where in the image is this object present.

For this case, fully convolutional networks (abbreviated as FCNs) are used. They operation called deconvolution is applied, to restore the spatial information. In this project, I have used VGG16 network and built three additional layers; each layer has 1x1 convolution followed by a deconvolution. The details of network are described in the paper mentioned above.

## Optimization
In this project, Adam Optimizer is used to minimize the cross-entropy loss with learning rate of 1e-5; the training was first done with learning rate of 1e-4

## Training with Augmented Images


## Comparison of Results with and without Augmentation
[um_000047.png]


## Saving and Restoring the Model


## Run on Videos


## Things to Look Forward to...
mean_iou training optimizer: https://arxiv.org/pdf/1608.01471.pdf
Cityscapes dataset
Better augmentation and hyperparameters tuning
Fusion