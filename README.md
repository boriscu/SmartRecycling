# Smart Recycling

This project aims to classify different types of waste materials using a deep learning model built with TensorFlow and Keras.

## Overview

The project uses a ResNet50 architecture pre-trained on ImageNet to classify waste materials into the following categories:
- Cardboard
- Glass bottle
- Can
- Crushed can
- Plastic bottle

## Dependencies

- numpy
- tensorflow
- matplotlib
- seaborn
- sklearn

## Dataset

The dataset used for this project can be downloaded from the following link:
[recycle_data_shuffled.tar.gz](http://web.cecs.pdx.edu/~singh/rcyc-web/recycle_data_shuffled.tar.gz)

After downloading, extract the dataset to get the training and testing data.

## Model Architecture

The model uses a ResNet50 base model with its layers frozen. On top of the base model, several dense layers are added for classification. The model is first trained without fine-tuning the base model. After the initial training, the last 10 layers of the base model are unfrozen and the model is fine-tuned.

## Data Augmentation

Data augmentation techniques such as rotation, width shift, height shift, shear, zoom, and horizontal flip are applied to the training data to improve the model's generalization.

## Training

The model is trained using the Adam optimizer with a learning rate schedule. Early stopping and model checkpointing are used as callbacks during training.

## Evaluation

After training, the model's performance is evaluated on the test data. The accuracy and loss plots for both the initial training and fine-tuning phases are provided. Additionally, a confusion matrix is plotted to visualize the model's performance across different classes.

## Prediction

The trained model can be used to predict the class of a given waste material image.
