# Plant Disease Detection Using CNN

## Overview

Automated plant disease detection using deep convolutional neural networks to help identify plant diseases from leaf images. Built with TensorFlow and Keras.

## Dataset

PlantVillage dataset from Kaggle: https://www.kaggle.com/sumanismcse/plant-disease-detection-using-keras

## Preprocessing

- Cropped images with 500+ pixel resolution
- Manual labeling with disease classifications
- Removed duplicates and low-quality images

## Model Architecture

![Network Architecture Overview](images/image2.png)

![Architecture Diagram](images/image3.png)

Deep CNN with:
- Convolutional layers with ReLU activation
- Batch normalization
- Max pooling
- Dropout
- Fully connected layers

## Results

**Accuracy: 96.77%**

![Training Results](images/image4.png)

Tested on validation set with individual class performance metrics.
