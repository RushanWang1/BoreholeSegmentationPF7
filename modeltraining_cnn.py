import numpy as np
from createDataset import dataset
import torch
from torch import nn
import evaluate
from transformers import TrainingArguments, Trainer
from torch.utils.data import WeightedRandomSampler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from transformers import default_data_collator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# create 'id2label'
id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation', 3: 'faultgauge', 4: 'breakout', 5: 'faultzone'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
train_ds = dataset["train"]
test_ds = dataset["validation"]

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(width=512, height=512, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    ToTensorV2()
])

def process_image(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    return image

# Processor function for transforming and preparing inputs
def processor_transform(images, labels):
    augmented_images = [augmentation_pipeline(image=process_image(img))['image'] for img in images]
    augmented_labels = [torch.tensor(process_image(lbl), dtype=torch.long) for lbl in labels]
    return augmented_images, augmented_labels

def train_transforms(example_batch):
    augmented_images, augmented_labels = processor_transform(example_batch['pixel_values'], example_batch['label'])
    return {'pixel_values': torch.stack(augmented_images), 'labels': torch.stack(augmented_labels)}

def val_transforms(example_batch):
    augmented_images, augmented_labels = processor_transform(example_batch['pixel_values'], example_batch['label'])
    # augmented_images = example_batch['pixel_values']
    # augmented_labels = example_batch['label']
    return {'pixel_values': torch.stack(augmented_images), 'labels': torch.stack(augmented_labels)}

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# Check availability of train and test dataset
print(train_ds, test_ds)

# Debugging: Check the first few items of the dataset
print("Training Data Sample:", train_ds[0])
print("Validation Data Sample:", test_ds[0])

# Load DeepLabV3 model
model = smp.DeepLabV3(encoder_name="resnet50", encoder_weights="imagenet", classes=num_labels, activation=None)
model = model.to(device)

# Weighted random sampler
weights = [1.4, 404.5, 143.1, 1619.2, 4.5, 13.8]
sample_weights = torch.tensor(weights, dtype=torch.float32)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_ds),
    replacement=True
)

# Training arguments
epochs = 400
lr = 0.00006
batch_size = 4

training_args = TrainingArguments(
    "DeepLabV3-ep400-batch4-augmentall-splitarea-512", 
    learning_rate=lr,
    dataloader_num_workers=4,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # fp16=True,
    evaluation_strategy="no",
    do_eval=False,
    save_total_limit=3,
    # evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    # eval_steps=20,
    logging_dir='DeepLabV3-ep400-batch4-augmentall-splitarea-512-log',
    logging_steps=1,
    eval_accumulation_steps=5,
    # load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory=True,
    use_cpu=False,    
)

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
        labels = inputs.get("labels").to(device)
        pixel_values = inputs.get("pixel_values").to(device, dtype=torch.float32)
        # forward pass
        outputs = model(pixel_values)
        logits = outputs  # The output is already a tensor
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=sample_weights.to(device))
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.tensor(logits).to(device)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor
        metrics = metric._compute(
                predictions=pred_labels.cpu(),
                references=labels.cpu(),
                num_labels=len(id2label),
                ignore_index=0,
                reduce_labels=False,
            )
        
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        
        return metrics

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    # eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
