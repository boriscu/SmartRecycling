# Smart Recycling

Smart Recycling is a project aimed at automating the process of identifying and localizing waste materials using deep learning. The project consists of two main components:

1. **Classification Model**: This model classifies the type of waste material.
2. **Localization Model**: This model detects and localizes the waste material in an image.

## Classification Model

The classification model is built using the ResNet50 architecture pre-trained on ImageNet. The model is fine-tuned on a dataset containing images of various waste materials.

### Dataset

The dataset contains images of the following waste materials:

- Cardboard
- Glass bottle
- Can
- Crushed can
- Plastic bottle

### Training

1. The ResNet50 base model is frozen, and custom dense layers are added on top.
2. The model is trained with data augmentation to increase its robustness.
3. After initial training, the last few layers of the ResNet50 base model are unfrozen, and the model is fine-tuned.

### Evaluation

The model's performance is evaluated using accuracy and loss metrics. A confusion matrix is also plotted to visualize the model's performance across different classes.

## Localization Model

The localization model uses the SSD MobileNet V2 architecture to detect and localize waste materials in images.

### Usage

Given an image, the model detects the waste material and draws a bounding box around it. If the detected waste material's bounding box has a square shape, it is directly classified. If not, the bounding box is resized to a square shape, and then the classification model classifies the waste material.

## Dependencies

- TensorFlow
- TensorFlow Hub
- Matplotlib
- Seaborn
- Scikit-learn
