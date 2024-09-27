import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from transformers import SegformerForSemanticSegmentation
from transformers import TrainerCallback, TrainingArguments
from createDataset import dataset
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, Compose, RandomCrop
from transformers import SegformerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer
from torch.utils.data import WeightedRandomSampler
# import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from FocalLoss import FocalLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# create 'id2label'
# id2label = {0: 'intactwall', 1: 'breakout', 2: 'faultzone', 3: 'wetspot', 4: 'unclassifycracks', 5: 'tectonictrace', 6: 'desiccation', 7: 'faultgauge'}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'incipientbreakout', 5: 'faultzone',6: 'fullybreakout'}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'inducedcrack',3: 'faultgouge', 4: 'breakout', 5: 'faultzone',6: 'breakoutinfaultzone',7:'inducedcrackinfaultzone'}
id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'breakout', 5: 'faultzone'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# One-hot map
one_hot_map = {
    0: [1, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0],
    4: [0, 0, 0, 0, 1, 0],
    5: [0, 0, 0, 0, 0, 1],
    6: [0, 0, 0, 0, 1, 1],
    7: [0, 0, 1, 0, 0, 1],
}


# load dataset
# trainarea_ds = dataset["train"]
# train_ds = dataset["train"].select(train_idxs)
# test_ds = dataset["train"].select(test_idxs)

train_test_split = dataset['train'].train_test_split(test_size=0.05, seed=42)

# Extracting the split datasets
train_ds = train_test_split['train']
test_ds = train_test_split['test']

# Image processor and augmentation
processor = SegformerImageProcessor(do_rescale= True, do_reduce_labels = False)
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(width=512, height=512, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),   
    A.CLAHE(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1), 
    # A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    # A.GridDistortion(p=0.5),
    # A.CLAHE(p=0.3)
])

def process_image(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    return image

def one_hot_encode(label, one_hot_map):
    num_classes = len(one_hot_map[0])
    one_hot = np.zeros((num_classes, *label.shape), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c, :, :] = np.isin(label, [key for key, val in one_hot_map.items() if val[c] == 1]).astype(np.uint8)
    return one_hot

# Processor function for transforming and preparing inputs
def processor_transform(images, labels):
    encodings = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return encodings

# def processor_transform(images, labels):
#     labels_one_hot = [one_hot_encode(label, one_hot_map) for label in labels]
#     encodings = processor(images=images, segmentation_maps=labels_one_hot, return_tensors="pt")
#     return encodings

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

train_ds = train_ds.shuffle(seed=42)
test_ds = test_ds.shuffle(seed=42)

# check availablity of train and test dataset
print(train_ds, test_ds)

# Fine-tune a SegFormer model
# pretrained_model_name = "nvidia/mit-b0" 
pretrained_model_name = '/home/wangrush/code/FineTune/model_ckpt/segformer_model/nowrs-ep500-batch24-512-1409/checkpoint-146000'
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    local_files_only=True,
    id2label=id2label,
    label2id=label2id
)


model = model.to(device)
#  set up trainer
# weighted random sampler
# class_counts = torch.tensor([238765416, 849106, 2401392, 212117, 76447632, 24784013], dtype=torch.float)

weights = [1.4,404.5,143.1,1619.2,4.5,13.8] # for 6class, without faultzone breakout and cracks
# weights = [1,404,143,1774,4,18,61,1609] # for 8class, with faultzone breakout and cracks
# sample_weights = torch.tensor(weights)
sample_weights = torch.zeros(len(train_ds))
sum_weights = sum(weights)
alpha = [w / sum_weights for w in weights]
# for idx, (pixel_values, label) in enumerate(train_ds):
#     class_weight = weights[label]
#     sample_weights[idx] = class_weight

####    Sampler Implementation Here!
# for i in range(len(train_ds)):
#     mask = train_ds[i]['labels']
#     for j in range(6):
#         if mask[j].sum() > 0:
#             sample_weights[i] += weights[j]

# epsilon = 1e-6
# sample_weights = sample_weights + epsilon

# sample_weights = sample_weights / sample_weights.sum()

# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(train_ds),
#     replacement=True
# )


# torch.cuda.set_device(0)
epochs = 500
lr = 0.0001
batch_size = 20

training_args = TrainingArguments(
    f"model_ckpt/segformer_model/nowrs-ep{epochs}-batch{batch_size}-multiscale-1709", 
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
    logging_dir = f'model_ckpt/segformer_model/nowrs-ep{epochs}-batch{batch_size}-multiscale-1709-log',
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
    # sampler=sampler,
    collate_fn=default_data_collator,
    num_workers=training_args.dataloader_num_workers,
    pin_memory=training_args.dataloader_pin_memory
)
class CustomTrainer(Trainer):
    # def get_train_dataloader(self):
    #     return train_dataloader  # Use the custom DataLoader created above
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        # loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device) # for integer label
        loss_fct = FocalLoss(alpha=torch.tensor(alpha).to(device), gamma=2, reduction='sum').to(device)  
        logits_tensor = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
            )
        labels = labels.float()
        loss = loss_fct(logits_tensor, labels.long())
        # loss = loss_fct(logits_tensor.reshape(-1, self.model.config.num_labels), labels.view(-1).long()) # for integer label
        return (loss, outputs) if return_outputs else loss



metric = evaluate.load("mean_iou")

def decode_predictions(predictions):
    """
    Decode the predictions from one-hot encoded format to original class labels.
    """
    decoded_preds = np.zeros((predictions.shape[0], predictions.shape[2], predictions.shape[3]), dtype=np.uint8)
    for i in range(predictions.shape[0]):
        for h in range(predictions.shape[2]):
            for w in range(predictions.shape[3]):
                one_hot_vector = predictions[i, :, h, w]
                for label, encoding in one_hot_map.items():
                    if np.array_equal(one_hot_vector, encoding):
                        decoded_preds[i, h, w] = label
                        break
    return decoded_preds

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
    
    # logits_tensor = nn.functional.interpolate(
    #         logits_tensor,
    #         size=labels.shape[-2:],
    #         mode="bilinear",
    #         align_corners=False,
    # ).sigmoid().round()
        
    # pred_labels = decode_predictions(logits_tensor.cpu().numpy())

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

class LogLossCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer
        self.train_loss = 0
        self.steps = 0

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        train_loss = logs.get("loss")
        if train_loss is not None:
            self.train_loss += train_loss
            self.steps += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        if self.steps > 0:
            avg_train_loss = self.train_loss / self.steps
            self.writer.add_scalar("Training Loss", avg_train_loss, epoch)
            self.train_loss = 0
            self.steps = 0

    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs.get("metrics", {})
        epoch = state.epoch
        val_loss = logs.get("eval_loss")
        if val_loss is not None:
            self.writer.add_scalar("Validation Loss", val_loss, epoch)
            
# print(dir(train_ds))
writer = SummaryWriter(log_dir=f'model_ckpt/segformer_model/nowrs-ep{epochs}-batch{batch_size}-augment-multiscale-evallog')
# Trainer => CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[LogLossCallback(writer)]
)
trainer.train()

