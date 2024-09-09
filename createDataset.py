import random
from datasets import Dataset, DatasetDict, Image, Features
import pandas as pd
from PIL import Image as PILImage
import numpy as np
import os
import csv
import cv2 

# # Number of files
total_files = 5103
# file_indices = list(range(total_files))
# # Shuffle the indices to ensure random distribution
# random.shuffle(file_indices)

# with open("file_indices.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for index in file_indices:
#         writer.writerow([index])
# file_indices = []
# with open("file_indices.csv", mode='r', newline='') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         if row:  # making sure the row is not empty
#             file_indices.append(int(row[0]))  # Convert string back to integer

# print("Shuffled indices have been loaded from the CSV file")
# print(shuffled_indices)

# Define the split point for 80% training, 20% testing
# split_index = int(total_files * 0.8)
# print(split_index)
# train_indices = file_indices[:split_index]
# test_indices = file_indices[split_index:]
# train_indices = range()

# Generate file paths for training and testing sets
train_size = 4158
test_size = 945
image_size = 512
# image_paths_train = [f'data/train_image_{image_size}/image_{i}.jpg' for i in range(train_size)]
# label_paths_train = [f'data/train_annotation_faultzone_{image_size}/annotation_{i}.png' for i in range(train_size)]
# image_paths_validation = [f'data/test_image_{image_size}/image_{i}.jpg' for i in range(test_size)]
# label_paths_validation = [f'data/test_annotation_faultzone_{image_size}/annotation_{i}.png' for i in range(test_size)]

image_paths_train_512 = [f'data/train_image_512/image_{i}.jpg' for i in range(4158)]
label_paths_train_512 = [f'data/train_annotation_512/annotation_{i}.png' for i in range(4158)]
image_paths_train_1024 = [f'data/train_image_1024/image_{i}.jpg' for i in range(1034)]
label_paths_train_1024 = [f'data/train_annotation_1024/annotation_{i}.png' for i in range(1034)]
image_paths_train_2048 = [f'data/train_image_2048/image_{i}.jpg' for i in range(235)]
label_paths_train_2048 = [f'data/train_annotation_2048/annotation_{i}.png' for i in range(235)]
image_paths_train_4096 = [f'data/train_image_4096/image_{i}.jpg' for i in range(46)]
label_paths_train_4096 = [f'data/train_annotation_4096/annotation_{i}.png' for i in range(46)]
image_paths_train = image_paths_train_512+image_paths_train_1024+image_paths_train_2048+image_paths_train_4096
label_paths_train = label_paths_train_512+label_paths_train_1024+label_paths_train_2048+label_paths_train_4096

image_paths_validation = [f'data/test_image_{image_size}/image_{i}.jpg' for i in range(test_size)]
label_paths_validation = [f'data/test_annotation_{image_size}/annotation_{i}.png' for i in range(test_size)]

# If you need to see the lists or do something with them, you can print them out or proceed with your logic
# print("Training image paths:", image_paths_train)
# print("Training label paths:", label_paths_train)
# print("Testing image paths:", image_paths_validation)
# print("Testing label paths:", label_paths_validation)


def create_dataset(image_paths, label_paths):
    
    dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset


# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict({
     "train": train_dataset,
     "validation": validation_dataset,
     }
)

# def dataset_create(image_paths_train, label_paths_train, image_paths_validation, label_paths_validation):
#     train_dataset = create_dataset(image_paths_train, label_paths_train)
#     validation_dataset = create_dataset(image_paths_validation, label_paths_validation)
#     dataset_new = DatasetDict({
#         "train": train_dataset,
#         "validation": validation_dataset,
#         }
#     )
#     return dataset_new
# dataset = dataset_create(image_paths_train, label_paths_train, image_paths_validation, label_paths_validation)

from torch.utils.data import Dataset as BaseDataset

# MyDataset for SegmentationModelTorch
# class MyDataset(BaseDataset):
#     """
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline 
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing 
#             (e.g. noralization, shape manipulation, etc.)
    
#     """
    
#     CLASSES = ['intactwall', 'tectonictrace', 'desiccation', 'faultgauge', 'breakout', 
#                'faultzone']
    
#     def __init__(
#             self, 
#             images_dir, 
#             masks_dir, 
#             classes=None, 
#             augmentation=None, 
#             preprocessing=None,
#     ):
#         self.ids = images_dir
#         # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#         self.images_fps = images_dir
#         self.masks_fps = masks_dir
#         # convert str names to class values on masks
#         # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#         # print("I am at mydataset!")
#         # print("images_dir", images_dir[1])
    
#     def __getitem__(self, i):
#         # print("i:",i)       

#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], 0)
#         # print("I am after cv2")
        
#         # # extract certain classes from mask (e.g. cars)
#         # masks = [(mask == v) for v in self.class_values]
#         # mask = np.stack(masks, axis=-1).astype('float')
        
#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
        
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
            
#         return image, mask
#         # return dict(image=image, mask=mask)

#     def __len__(self):
#         return len(self.ids)





def color_palette():
    """Color palette that maps each class to RGB values.
    
    This one is actually taken from ADE20k.
    """
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

palette = color_palette()

