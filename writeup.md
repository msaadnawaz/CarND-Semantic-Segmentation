# Semantic Segmentation
The objective of this project is to find the free space on road using a deep learning technique called Fully Convolutional Networks (link:). The dataset used for training of deep neural network is KITTI dataset (link:) which has two classes i.e. road area and non-road area. 

## Fully Convolutinal Networks
The CNNs are good for classification but they los the spatial information which means that the CNNs can tell if there is specific object present in image or not (when trained to classify the object) but can't tell where in the image is this object present.

For this case, fully convolutional networks (abbreviated as FCNs) are used. They operation called deconvolution is applied, to restore the spatial information. In this project, I have used VGG16 network and built three additional layers; each layer has 1x1 convolution followed by a deconvolution. The details of network are described in the paper mentioned above.

## Optimization
In this project, Adam Optimizer is used to minimize the cross-entropy loss with learning rate of 1e-5; the training was first done with learning rate of 1e-4 but citing fluctuations in loss over epochs, it was decided to reduce the learning rate

## Training with Augmented Images
The training of this fully convolutional network was done on KITTI dataset with two classes: road area (free space) and non-road area. However, KITTI dataset is very small to create a generalized model capable of detecting free-space on any image/video clip, therefore, the augmented set was appended additionally to the training data. The augmented data was constructed by flipping every image horizontally, then image translation was applied on half of images picked randomly. Similarly, 50% of images were picked randomly and their brightness was tweaked to train model on different brightness levels.

While training, dropout of 50% was applied on unaugmented images while only 25% were kept from the augmented set.

## Comparison of Results with and without Augmentation
Augmentation convincingly improved the performance and converted blur edges of free-space in sharp edges. Example shown here:
[um_000047.png]


## Saving and Restoring the Model
It took around half an hour to train the model for 20 epochs on AWS p2-xlarge machine (with Nvidia Tesla K80 GPU) and the model was stored using the utility provided by tensorflow. Similarly, it was restored to run on videos on local PC.

## Run on Videos
After restoring, the model was used to test the videos from different sources, however, the results were not convincing enough. The major reason being insufficient distinct data for training. The video outputs can be seen in ./runs/videos_output folder.

## Things to Look Forward to...
To improve the performance, the following changes can be employed:
	
	1. Minimization of cross-entropy loss is not an ideal metric to learn on, for FCNs because each batch may contain different types of images and calculation of loss over whole image may yield fluctuations. Instead, a differentiable form of mean IOU can be used as training optimizer as mentioned here: https://arxiv.org/pdf/1608.01471.pdf
	2. As mentioned earlier, KITTI dataset is insufficient for creating a generalized model hence any other more comprehensive dataset like Cityscapes dataset can be used to train the network to expect better results.
	3. More augmentation followed by hyperparameters tuning accordingly can also yield better results.
	4. Fusion techniques can be employed to reduce the time for training as well as memory footprint.