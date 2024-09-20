## Advanced fracture identification in borehole imagery through semantic segmentation
This project focuses on the application of semantic segmentation techniques to automatically identify six features in borehole imagery. 
By leveraging advanced machine learning models such as U-Net and SegFormer, the goal is to enhance the accuracy and efficiency of fracture detection, which plays a critical role in geological analysis.


## Setup
To run the code, you'll need Python 3. We recommend using a virtual environment to manage dependencies. You can set up the environment and install the required packages as follows:
```
python3 -m venv borehole_venv
source borehole_venv/bin/activate
pip install -r requirements.txt
```

## Overview of the code
```utils.py```
This file contains key utility classes and functions:
- FocalLoss: A custom loss function designed to address class imbalance during training
- UnetModel: A custom class implementation of the U-Net architecture

```dataprocess.py```
Handles the creation of datasets for training and testing:
- The image_size argument determines how the borehole image is cropped into smaller segments
- If with_test_set is set to True, images "230816" and "210202" are divided into a training set (75%) and a test set (25%). If set to False, all images are used for training without creating a separate test set.  

```unet_training.py```
Used for training a U-Net model and for testing different hyperparamtera. The following arguments are configurable:
- num_workers: Number of workers for data loading.
- epochs: Number of training epochs.
- encoder_name: Type of encoder backbone used in the U-Net model.
- val_split: Ratio of data used for validation.
- random_seed: Seed to ensure reproducibility.
- image_size: Specifies the size of the cropped image segments.
- augmented: If True, applies data augmentation.
- initial_learning_rate: Initial learning rate for the optimizer.
- loss_function: The loss function to use (e.g., FocalLoss).
- batch_size_train_val: Batch size for training and validation.
- batch_size_test: Batch size for testing.

```unet_testing.py```
This script loads a pre-trained model checkpoint and tests it on a full-sized borehole image. The process involves:
- Predicting on cropped segments of the image.
- Stitching the predicted segments together to reconstruct the full image.
- Calculating performance metrics on the entire borehole image to evaluate model accuracy.
- Creates files that can be used to visualize the prediction

```segformer_training.py```
(In Progress)
This script is intended for training a SegFormer model, but it is not yet complete.

```visualize.ipynb```
A Jupyter notebook designed for visualizing both the borehole images and their corresponding annotations.


