from torch import nn
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from createDataset import dataset
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# create 'id2label'
id2label = {0: 'intactwall', 1: 'breakout', 2: 'faultzone', 3: 'wetspot', 4: 'unclassifycracks', 5: 'tectonictrace', 6: 'desiccation', 7: 'faultgauge'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
train_ds = dataset["train"]
test_ds = dataset["validation"]

# Image processor and augmentation
processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path='/home/wangrush/code/FineTune/segformer-b0-finetuned-ep50/checkpoint-25540', local_files_only=True)

m_accuracy=list()
mean_IOU = list()
perClass_IOU = np.arange(8)

def mIOU(prediction, label, num_classes):
    # prediction= prediction.max(1)[1].float().cpu().numpy()
    # label = label.float().cpu().numpy() 
    
    iou_list = list()
    present_iou_list = list()
    # all_iou_list = list()

    for sem_class in range(num_classes):
        # print(sem_class)
        pred_inds = (prediction == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
            # iou_now = 0
            # all_iou_list
        else:
            # print(sem_class)
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    miou = np.mean(present_iou_list)
    return miou, iou_list

for n in range(245):
    # print(n)
    image = test_ds[n]['pixel_values']
    gt_seg = test_ds[n]['label']
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (height, width)
    mode='bilinear',
    align_corners=False
    )
    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_arr = np.array(pred_seg)
    gt_arr = np.array(gt_seg)
    acc = np.mean((pred_arr == gt_arr))
    m_accuracy.append(acc)
    m_iou,iou_list = mIOU(pred_arr,gt_arr,8)
    iou_arr = np.array(iou_list)
    mean_IOU.append(m_iou)
    # print(type(perClass_IOU))
    perClass_IOU = np.vstack([perClass_IOU,iou_arr])