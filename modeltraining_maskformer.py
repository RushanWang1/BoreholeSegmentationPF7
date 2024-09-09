import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from transformers import MaskFormerForInstanceSegmentation
from transformers import TrainingArguments
from createDataset import dataset
# from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, Compose, RandomCrop
from transformers import MaskFormerImageProcessor
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate
from transformers import Trainer
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# create 'id2label'
# id2label = {0: 'intactwall', 1: 'breakout', 2: 'faultzone', 3: 'wetspot', 4: 'unclassifycracks', 5: 'tectonictrace', 6: 'desiccation', 7: 'faultgauge'}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'incipientbreakout', 5: 'faultzone',6: 'fullybreakout'}
# id2label = {0: 'intactwall', 1: 'tectonictrace', 2: 'desiccation',3: 'faultgauge', 4: 'breakout', 5: 'faultzone',}
id2label = {0: 'background', 1: 'intactwall', 2: 'tectonictrace', 3: 'desiccation',4: 'faultgauge', 5: 'breakout', 6: 'faultzone',}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# load dataset
train_ds = dataset["train"]
test_ds = dataset["validation"]

# Image processor and augmentation  
processor = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)  
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(width=512, height=512, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    # A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
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

train_transform = augmentation_pipeline

test_transform = A.Compose([
    A.Resize(width=512, height=512),
    # A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])

# def train_transforms(example_batch):
#     augmented_images = []
#     augmented_labels = []
    
#     for image, label in zip(example_batch['pixel_values'], example_batch['label']):
#         # Convert PIL images to numpy arrays
#         image = process_image(image)
#         label = process_image(label)
        
#         # Apply the augmentation
#         augmented = augmentation_pipeline(image=image, mask=label)
#         augmented_images.append(augmented['image'])
#         augmented_labels.append(augmented['mask'])
    
#     inputs = processor_transform(augmented_images, augmented_labels)
#     return inputs

# def val_transforms(example_batch):
#     images = [process_image(image) for image in example_batch['pixel_values']]
#     labels = [process_image(label) for label in example_batch['label']]
#     inputs = processor_transform(images, labels)
#     return inputs

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]['pixel_values'])
        original_segmentation_map = np.array(self.dataset[idx]['label'])
        
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2,0,1)
        segmentation_map = segmentation_map+1

        return image, segmentation_map, original_image, original_segmentation_map


# Set transforms
# train_ds.set_transform(train_transforms)
# test_ds.set_transform(val_transforms)

train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)


# check availablity of train and test dataset
print(train_ds, test_ds)

# Fine-tune a Maskformer model 
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)



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
epochs = 100
lr = 0.00006
batch_size = 4

training_args = TrainingArguments(
    "Maskformer-ep100-batch4-allaugment-splitarea-512", 
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
    logging_dir = 'Maskformer-ep100-batch4-allaugment-splitarea-512-log',
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory = True,
    use_cpu= False,    
)

from torch.utils.data import DataLoader
from transformers import default_data_collator
def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # if sum(sum(segmentation_maps[0])) == 0 or sum(sum(segmentation_maps[1])) ==0:
    #     return 
    batch = processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    # batch["original_images"] = inputs[2]
    # batch["original_segmentation_maps"] = inputs[3]
    
    return batch

train_dataloader = DataLoader(
    train_dataset,  # Your training dataset
    batch_size=training_args.per_device_train_batch_size,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=training_args.dataloader_num_workers,
    pin_memory=training_args.dataloader_pin_memory
)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# class CustomTrainer(Trainer):
#     def get_train_dataloader(self):
#         return train_dataloader  # Use the custom DataLoader created above
    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         # compute custom loss
#         loss_fct = nn.CrossEntropyLoss(weight=sample_weights)# 
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss



metric = evaluate.load("mean_iou")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
writer = SummaryWriter(log_dir=training_args.logging_dir)

for epoch in range(epochs):
    print("Epoch:", epoch)
    model.train()
    
    running_loss = 0.0
    num_samples = 0
    
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device).to(torch.float),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        # Optimization
        optimizer.step()
        
        # Log training loss every 100 iterations
        if idx % 100 == 0:
            writer.add_scalar('Training Loss', running_loss / num_samples, epoch * len(train_dataloader) + idx)
            print("Loss:", running_loss / num_samples)
    
    # Log average training loss for the epoch
    avg_train_loss = running_loss / num_samples
    writer.add_scalar('Average Training Loss', avg_train_loss, epoch)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_num_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device).to(torch.float),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            loss = outputs.loss
            batch_size = batch["pixel_values"].size(0)
            val_running_loss += loss.item() * batch_size
            val_num_samples += batch_size

    # Log validation loss for the epoch
    avg_val_loss = val_running_loss / val_num_samples
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    print("Validation Loss:", avg_val_loss)
    
    # Calculate and log Mean IoU
    for idx, batch in enumerate(tqdm(test_dataloader)):
        if idx > 5:
            break

        pixel_values = batch["pixel_values"]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values.to(device))

        # get original images
        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        # predict segmentation maps
        predicted_segmentation_maps = processor.post_process_semantic_segmentation(outputs,
                                                                                   target_sizes=target_sizes)

        # get ground truth segmentation maps
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]

        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
    
    # Log Mean IoU
    mean_iou = metric.compute(num_labels=len(id2label), ignore_index=0)['mean_iou']
    writer.add_scalar('Mean IoU', mean_iou, epoch)
    print("Mean IoU:", mean_iou)

torch.save(model, 'Maskformer-ep100-batch4-allaugment-splitarea-512.pt') 
# Close the SummaryWriter
writer.close() 


# def compute_metrics(eval_pred):
#   with torch.no_grad():
#     logits, labels = eval_pred
#     logits_tensor = torch.from_numpy(logits)
#     # scale the logits to the size of the label
#     logits_tensor = nn.functional.interpolate(
#         logits_tensor,
#         size=labels.shape[-2:],
#         mode="bilinear",
#         align_corners=False,
#     ).argmax(dim=1)

#     pred_labels = logits_tensor# .detach().cpu().numpy()
#     # currently using _compute instead of compute
#     # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
#     metrics = metric._compute(
#             predictions=pred_labels,
#             references=labels,
#             num_labels=len(id2label),
#             ignore_index=0,
#             reduce_labels=processor.do_reduce_labels,
#         )
    
#     # add per category metrics as individual key-value pairs
#     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
#     per_category_iou = metrics.pop("per_category_iou").tolist()

#     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
#     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
#     return metrics
  
# # print
# # print(dir(train_ds))
# # Trainer => CustomTrainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=test_ds,
#     compute_metrics=compute_metrics,
# )
# trainer.train()

