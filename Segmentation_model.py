import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import segmentation_models_pytorch.utils

from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from createDataset import dataset, image_paths_train, label_paths_train, image_paths_validation, label_paths_validation
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, Compose
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
from torch import nn
import evaluate
from transformers import Trainer
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from createDataset import MyDataset

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'breakout', 5: 'faultzone',}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
train_ds = dataset["train"]
test_ds = dataset["validation"]

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        img = image.T
        plt.imshow(img)
    plt.show()


# Dataloader HERE
x_train_dir = image_paths_train
y_train_dir = label_paths_train
x_valid_dir = image_paths_validation
y_valid_dir = label_paths_validation

dataset = MyDataset(x_train_dir, y_train_dir, classes=['intactwall', 'tectonictrace', 'desiccation', 'faultgauge', 'breakout', 'faultzone'])

# image, mask = dataset[2] # get some sample
# visualize(
#     image=image, 
#     cars_mask=mask.squeeze(),
# )

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.augmentations.transforms.GaussNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                # albu.augmentations.transforms.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.augmentations.transforms.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, ),
    ]
    return albu.Compose(_transform)

#### Visualize resulted augmented images and masks

# augmented_dataset = MyDataset(
#     x_train_dir, 
#     y_train_dir, 
#     augmentation=get_training_augmentation(), 
#     # classes=['car'],
# )

# same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze())


print("Successfully did the dataloading!")


ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['intactwall', 'tectonictrace', 'desiccation', 'faultgauge', 'breakout', 'faultzone']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    in_channels=3,  
    classes=num_labels, 
    activation=ACTIVATION,
    # device = DEVICE
)
model = model.to(device)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = MyDataset(
    x_train_dir, 
    y_train_dir, 
    # augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
# image, mask = train_dataset[2] # get some sample
# visualize(
#     image=image, 
#     cars_mask=mask.squeeze(),
# )

valid_dataset = MyDataset(
    x_valid_dir, 
    y_valid_dir, 
    # augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
weights = [1.4,404.5,143.1,1619.2,4.5,13.8]
sample_weights = torch.tensor(weights)

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=4082,
    replacement=True
)
# batch_sampler_train = torch.utils.data.BatchSampler(weighted_sampler, 
#                                                     batch_size=12, 
#                                                     drop_last=True)

train_loader = DataLoader(train_dataset, batch_size=1, sampler=weighted_sampler,num_workers=4,pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True)


loss_function = segmentation_models_pytorch.utils.losses.DiceLoss()
# loss_function = segmentation_models_pytorch.utils.losses.WeightedCrossEntropyLoss(class_weights=weights)
loss_function = nn.CrossEntropyLoss(weight=sample_weights).to(device)

metrics = [
    segmentation_models_pytorch.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.00006),
])

# create epoch runners 
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs_logits = model(inputs)

        outputs = torch.argmax(outputs_logits, axis = 1)
        outputs_logits = outputs_logits.to(device)

        # Compute the loss and its gradients
        loss = loss_function(outputs_logits, labels.long())
        loss.requires_grad = True
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return running_loss # last_loss
# it is a simple loop of iterating over dataloader`s samples
# train_epoch = segmentation_models_pytorch.utils.train.TrainEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )

# valid_epoch = segmentation_models_pytorch.utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )

# train model for 50 epochs
epochs_num = 50
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

max_score = 0
for epoch in range(epochs_num):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(valid_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs_logit = model(vinputs)
            voutputs = torch.argmax(voutputs_logit, axis = 1)
            vloss = loss_function(voutputs, vlabels)
            print(vloss)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state< best_vloss
    # if avg_vloss :
    #     best_vloss = avg_vloss
    #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)

    epoch_number += 1

model_path = 'UnetModel/model_{}_{}'.format(timestamp, epoch_number)
torch.save(model, model_path)


model.eval()


# for i in range(0, epochs_num):
    
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
    
#     # do something (save model, change lr, etc.)
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model, './best_model.pth')
#         print('Model saved!')
        
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')
