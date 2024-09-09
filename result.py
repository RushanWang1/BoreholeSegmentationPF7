from datasets import Dataset, DatasetDict, Image, Features
import pandas as pd
from PIL import Image as PILImage
import numpy as np
from torch import nn
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
# from createDataset import dataset
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
from PIL import Image

# train_size = 4158
# test_size = 945
train_size = 1034
test_size = 188
image_size = 1024

# Generate file paths for training and testing sets
# image_paths_train = [f'data/train_image_{image_size}/image_{i}.jpg' for i in range(train_size)]
# label_paths_train = [f'data/train_annotation_{image_size}/annotation_{i}.png' for i in range(train_size)]
# image_paths_validation = [f'data/test_image_{image_size}/image_{i}.jpg' for i in range(test_size)]
# label_paths_validation = [f'data/test_annotation_{image_size}/annotation_{i}.png' for i in range(test_size)]

# def create_dataset(image_paths, label_paths):
    
#     dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
#                                 "label": sorted(label_paths)})
#     dataset = dataset.cast_column("pixel_values", Image())
#     dataset = dataset.cast_column("label", Image())
#     return dataset


# # step 1: create Dataset objects
# train_dataset = create_dataset(image_paths_train, label_paths_train)
# validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# # step 2: create DatasetDict
# dataset = DatasetDict({
#      "train": train_dataset,
#      "validation": validation_dataset,
#      }
# )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# create 'id2label'
# id2label = {0: 'intactwall', 1: 'breakout', 2: 'faultzone', 3: 'wetspot', 4: 'unclassifycracks', 5: 'tectonictrace', 6: 'desiccation', 7: 'faultgauge'}
id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'breakout', 5: 'faultzone',}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'incipientbreakout', 5: 'faultzone',6:'fullybreakout'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
# train_ds = dataset["train"]
# test_ds = dataset["validation"]

# Image processor and augmentation
processor = SegformerImageProcessor()
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

# def train_transforms(example_batch):
#     images = [jitter(x) for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = processor(images, labels)
#     return inputs


# def val_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = processor(images, labels)
#     return inputs


# # Set transforms
# train_ds.set_transform(train_transforms)
# test_ds.set_transform(val_transforms)

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path='Segformer-mytrainer-onehot-noweight-ep100-batch12-lr0001-augment-splitarea-512/checkpoint-34700', local_files_only=True)


allresult = np.zeros((6, 7091,48436))
width = 512*2
height = 512*2
num_image = 392 # 1485 
col_num = 48436 //(width-20) +1
row_num = 7091//(height-20) +1
horz_flip = True
for i in range(num_image):
    # image = test_ds_orig[n]['pixel_values']
    # gt_seg = test_ds_orig[n]['labels']
    image = Image.open('data/temporal_compare_data/201113_image_1024_overlap_20/' + f'image_{i}.jpg')
    # gt_seg = Image.open('data/annotation_overlap/' + f'annotation_{i}.png')
    if horz_flip:
        horz_flip_img = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        image = horz_flip_img
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    # print(logits.shape)
    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode='bilinear',
        align_corners=False
    )
    probability = nn.functional.softmax(upsampled_logits, dim = 1)
    # Second, apply argmax on the class dimension
    # pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_arr = probability.detach().numpy()  
    if horz_flip:
        horz_flip_img = pred_arr[:,::-1]
        pred_arr = horz_flip_img
    row_idx = i//col_num
    col_idx = i%col_num
    # print("Current pred unique and shape: ",np.unique(pred_arr))
    # print(row_idx, col_idx,pred_arr.shape)
    if col_idx<(col_num-1):
        col = col_idx*(width-20)          
    elif col_idx==(col_num-1):
        col = 48436 - width                 
    if row_idx<(row_num-1):
        row = row_idx*(height-20)
    elif row_idx==(row_num-1):
        row = 7091 - height
    if col_idx == 0 or col_idx ==(col_num-1) or row_idx ==0 or row_idx ==(row_num-1):
        allresult[:,int(row):int(row)+height, int(col):int(col)+width] = pred_arr[0]
    else:
        allresult[:,int(row)+10:int(row)+height-10, int(col) +10:int(col)+width-10] = pred_arr[0,:,10:-10,10:-10]
        
print(allresult.shape)       
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_intactwall.png', allresult[0])
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_tectonictrace.png', allresult[1])
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_desiccation.png', allresult[2])
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_faultgauge.png', allresult[3])
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_breakout.png', allresult[4])
plt.imsave(f'data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_faultzone.png', allresult[5])

with rasterio.open("data/refined/RadiusCropped.tif") as radiusimg:
    profile = radiusimg.profile

with rasterio.open('data/temporal_compare_data/maskprediction_201113/201113_prediction_augment_noweightloss_nowrs_1024_splitarea_512_intactwall.tif', 'w', **profile) as dst:
    dst.write(allresult[0], 1)


print("Finish!")    
