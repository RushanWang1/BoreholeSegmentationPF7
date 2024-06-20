import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from createDataset import dataset
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, Compose, RandomCrop
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer
from torch.utils.data import WeightedRandomSampler
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# create 'id2label'
# id2label = {0: 'intactwall', 1: 'breakout', 2: 'faultzone', 3: 'wetspot', 4: 'unclassifycracks', 5: 'tectonictrace', 6: 'desiccation', 7: 'faultgauge'}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'incipientbreakout', 5: 'faultzone',6: 'fullybreakout'}
id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'breakout', 5: 'faultzone',}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
train_ds = dataset["train"]
test_ds = dataset["validation"]

# Image processor and augmentation
processor = SegformerImageProcessor(do_rescale= True)
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(width=512, height=512, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    # A.GridDistortion(p=0.5),
    # A.CLAHE(p=0.5)
])

def process_image(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    return image

# Processor function for transforming and preparing inputs
def processor_transform(images, labels):
    encodings = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return encodings

def train_transforms(example_batch):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(example_batch['pixel_values'], example_batch['label']):
        # Convert PIL images to numpy arrays
        image = process_image(image)
        label = process_image(label)
        
        # Apply the augmentation
        augmented = augmentation_pipeline(image=image, mask=label)
        augmented_images.append(augmented['image'])
        augmented_labels.append(augmented['mask'])
    
    inputs = processor_transform(augmented_images, augmented_labels)
    return inputs

def val_transforms(example_batch):
    images = [process_image(image) for image in example_batch['pixel_values']]
    labels = [process_image(label) for label in example_batch['label']]
    inputs = processor_transform(images, labels)
    return inputs

# def train_transforms(example_batch):
#     images_1 = [augmentation_pipeline(x) for x in example_batch['pixel_values']]
#     labels_1 = [augmentation_pipeline(x) for x in example_batch['label']]
#     images = [jitter(x) for x in images_1]
#     labels = [x for x in labels_1]
#     # images = [jitter(x) for x in example_batch['pixel_values']]
#     # labels = [x for x in example_batch['label']]
#     # labels = np.where(labels == 5, 1, 0)
#     inputs = processor(images, labels)
#     return inputs


# def val_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = processor(images, labels)
#     return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


# check availablity of train and test dataset
print(train_ds, test_ds)

# Fine-tune a SegFormer model
pretrained_model_name = "nvidia/mit-b0" 
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

# # smp
# ENCODER = 'resnet34'
# ENCODER_WEIGHTS = 'imagenet'
# # CLASSES = ['car']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
# DEVICE = 'cuda'

# # create segmentation model with pretrained encoder
# model = smp.Unet(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     in_channels = 3,
#     classes=num_labels, 
#     activation=ACTIVATION,
# )


model = model.to(device)
#  set up trainer
# weighted random sampler
weights = [1.4,404.5,143.1,1619.2,4.5,13.8]
sample_weights = torch.tensor(weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_ds),
    replacement=True
)

# torch.cuda.set_device(0)
epochs = 600
lr = 0.00006
batch_size = 24

training_args = TrainingArguments(
    "Segformer-ep600-batch20-augmentall-splitarea-1024",
    learning_rate=lr,
    dataloader_num_workers= 4,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_dir = 'Segformer-ep600-batch20-augmentall-splitarea-1024-log',
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory = True,
    use_cpu= False,    
)

from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataloader = DataLoader(
    train_ds,  # Your training dataset
    batch_size=training_args.per_device_train_batch_size,
    sampler=sampler,
    collate_fn=default_data_collator,
    num_workers=training_args.dataloader_num_workers,
    pin_memory=training_args.dataloader_pin_memory
)
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return train_dataloader  # Use the custom DataLoader created above
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=sample_weights)# 
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor# .detach().cpu().numpy()
    # currently using _compute instead of compute
    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=processor.do_reduce_labels,
        )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics
  
# print
# print(dir(train_ds))
# Trainer => CustomTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

