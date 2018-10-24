# Kaggle-RSNA-Pneumonia-Detection-Challenge
build an algorithm that automatically detects potential pneumonia cases [Kaggle-RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)</br>
This repository is used for recording codes of Kaggle-RSNA-Pneumonia-Detection-Challenge.</br>

* ### 1. Problem Description
The aim of this competition is detecting Pneumonia. In the formal description, it means the Lung Opacity. However, among the datasets, there are three kinds of classes. They are ```Lung Opacity```, ```Normal```, ```No Lung Opacity/AbNormal```.</br>
So, I splited it into two stages. First is classification of Lung Opacity as a **binary classification problem**. Second is detecting Lung Opacity. Here I adopted **semantic segmentation**.</br>
* ### 2. Work Flow
  * #### 2.1 Classifier Network
  Here I adopted VGG16 as the basic framework for classification. Meanwhile, I added some tricks in it, such as **Dense Blocks** and so on.</br>
   **Dense Block** is first proposed in **CVPR2017** [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf). It's inspired from **ResNet**. In ResNet, authors add a skip-connection that bypasses the non-linear transformations with an identity function:</br>
   ![img](https://github.com/mjDelta/Kaggle-RSNA-Pneumonia-Detection-Challenge/blob/master/imgs/resnet.png)</br>
While in **Dense Block**, authors introduce **direct connections from any layer to all subsequent layers**.</br>
   ![img](https://github.com/mjDelta/Kaggle-RSNA-Pneumonia-Detection-Challenge/blob/master/imgs/denseblock.png)</br>
And it's function description is here.</br>
   ![img](https://github.com/mjDelta/Kaggle-RSNA-Pneumonia-Detection-Challenge/blob/master/imgs/densenet.png).</br>
So, the advantages of Dense Block(more direct connections) is:</br>
  * More efficiency useage of feature maps
  * Easier gradient update, especially for gradient vanishing problem.
  * #### 2.2 Semantic Segmentation Network
  
