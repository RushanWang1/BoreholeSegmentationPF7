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
train_ds = dataset["train"]
test_ds = dataset["validation"]
# train_ds, test_ds = train_ds_all.train_test_split(test_size=0.1, seed=42)

# Image processor and augmentation
processor = SegformerImageProcessor(do_rescale= True)

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
    A.CLAHE(p=0.2)
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
# def processor_transform(images, labels):
#     encodings = processor(images=images, segmentation_maps=labels, return_tensors="pt")
#     return encodings

def processor_transform(images, labels):
    encodings = processor(images=images, return_tensors="pt")
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
    label_array = np.array(augmented_labels)
    label_tensor = torch.from_numpy(label_array)
    inputs['original_labels'] = label_tensor.type(torch.LongTensor)
    return inputs

def val_transforms(example_batch):
    images = [process_image(image) for image in example_batch['pixel_values']]
    labels = [process_image(label) for label in example_batch['label']]
    inputs = processor_transform(images, labels)
    return inputs

# Custom Dataset Class
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, label = self.dataset[idx]['pixel_values'], self.dataset[idx]['label']
#         image = process_image(image)
#         label = process_image(label)
        
#         if self.transform:
#             augmented = self.transform(image=image, mask=label)
#             image = augmented['image']
#             label = augmented['mask']
        
#         label_one_hot = one_hot_encode(label, one_hot_map)
#         encodings = processor(images=[image], segmentation_maps=[label_one_hot], return_tensors="pt")
        
#         return {key: encodings[key][0] for key in encodings}

# # Initialize CustomDataset
# train_ds = CustomDataset(dataset["train"], transform=augmentation_pipeline)
# test_ds = CustomDataset(dataset["validation"], transform=None)

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


# check availablity of train and test dataset
print(train_ds, test_ds)

# Fine-tune a SegFormer model
pretrained_model_name = "nvidia/mit-b0"   # pretrained_model_name = "/home/wangrush/code/FineTune/Segformer-ep800-batch20-augmentall-splitarea-multiscale/checkpoint-99400"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    # local_files_only=True,
    id2label=id2label,
    label2id=label2id
)

model = model.to(device)

# from transformers import SegformerForSemanticSegmentation, SegformerConfig

# class CustomSegformer(SegformerForSemanticSegmentation):
#     def __init__(self, config):
#         super().__init__(config)
#         # Change the classifier to output 6 channels instead of 8
#         self.decode_head.classifier = nn.Conv2d(config.hidden_sizes[-1], 6, kernel_size=(1, 1))

# # Load the pretrained configuration
# config = SegformerConfig.from_pretrained(pretrained_model_name, num_labels=8)
# # Modify the configuration to have 6 output channels
# config.num_labels = 6

# # Instantiate the custom model with the modified configuration
# model = CustomSegformer(config)
# model = model.to(device)

#  set up trainer
# weighted random sampler
# weights = [1.4,404.5,143.1,1619.2,4.5,13.8] # for 6class, without faultzone breakout and cracks
# # weights = [1,404,143,1774,4,18,61,1609] # for 8class, with faultzone breakout and cracks
# sample_weights = torch.tensor(weights)
class_counts = torch.tensor([238765416, 849106, 2401392, 212117, 76447632, 24784013], dtype=torch.float)

# Calculate pos_weight for BCEWithLogitsLoss
total_pixels = class_counts.sum().item()
pos_weight = (total_pixels - class_counts) / class_counts
# pos_weight = total_pixels/class_counts
print("Pos Weight:", pos_weight)

sample_weights = torch.zeros(len(train_ds))
# for i in range(len(train_ds)):
#     sample_weights[i] = sum([pos_weight[j] for j in range(6) if train_ds[i][1][j].sum() > 0])

# # Normalize sample weights
# sample_weights = sample_weights / sample_weights.sum()

# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(train_ds),
#     replacement=True
# )

# torch.cuda.set_device(0)
epochs = 100
lr = 0.0001
batch_size = 12

training_args = TrainingArguments(
    "Segformer-mytrainer-onehot-bcelossweight-ep100-batch12-lr0001-augment-splitarea-512", 
    learning_rate=lr,
    dataloader_num_workers= 4,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    fp16=True,
    save_total_limit=3,
    # evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    # eval_steps=20,
    logging_dir = 'Segformer-mytrainer-onehot-bcelossweight-ep100-batch12-lr0001-augment-splitarea-512-log',
    logging_steps=1,
    # eval_accumulation_steps=5,
    # load_best_model_at_end=True,
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

# weight_loss = torch.tensor([0.01,0.18,0.06,0.74,0.01,0.01])
# weight_loss = weight_loss.view(1,6,1,1).to(device)
# weight_pos = torch.tensor([0.44,403,142,1618,3.5,12.9])
# pos_weights = weight_pos.view(6, 1, 1)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return train_dataloader  # Use the custom DataLoader created above
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("original_labels").to(device)
        labels_one_hot = [one_hot_encode(label.detach().cpu().numpy(), one_hot_map) for label in labels]
        labels_one_hot_arr=np.array(labels_one_hot)
        # forward pass
        input_onlyimage = {}
        input_onlyimage['pixel_values'] = inputs['pixel_values']
        outputs = model(**input_onlyimage)
        logits = outputs.get('logits')
        # compute custom loss // pos_weight=torch.FloatTensor(weights) pos_weight=pos_weight.to(device)
        loss_fct = nn.BCEWithLogitsLoss(reduction='none').to(device)
        # loss_fct = nn.L1Loss().to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=sample_weights).to(device) # for integer label
        logits_tensor = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
            )
        # probability = nn.functional.softmax(logits_tensor, dim = 1)
        # labels = labels.float()
        gt = torch.from_numpy(labels_one_hot_arr).to(device)
        loss_perclass = pos_weight.to(device)
        loss_all = loss_fct(logits_tensor, gt.float()).to(device)
        loss_perclass[0] = loss_all[:,0].mean()
        loss_perclass[1] = loss_all[:,1].mean()
        loss_perclass[2] = loss_all[:,2].mean()
        loss_perclass[3] = loss_all[:,3].mean()
        loss_perclass[4] = loss_all[:,4].mean()
        loss_perclass[5] = loss_all[:,5].mean()

        # loss_perclass[1] = loss_fct(logits_tensor[:,1], gt[:,1].float()).to(device)
        # loss_perclass[2] = loss_fct(logits_tensor[:,2], gt[:,2].float()).to(device)
        # loss_perclass[3] = loss_fct(logits_tensor[:,3], gt[:,3].float()).to(device)
        # loss_perclass[4] = loss_fct(logits_tensor[:,4], gt[:,4].float()).to(device)
        # loss_perclass[5] = loss_fct(logits_tensor[:,5], gt[:,5].float()).to(device)
        # loss = torch.matmul(loss_perclass, pos_weight.to(device))
        # loss = torch.sum(loss_perclass)
        # loss = loss_fct(logits_tensor, gt.float()).to(device)
        loss = (loss_perclass * pos_weight.to(device)).mean()
        # loss_weighted = loss*weight_loss
        # loss_weighted = loss_weighted.mean()
        # loss = loss_fct(logits_tensor.reshape(-1, self.model.config.num_labels), labels.view(-1)) # for integer label
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
    # logits_tensor = nn.functional.interpolate(
    #     logits_tensor,
    #     size=labels.shape[-2:],
    #     mode="bilinear",
    #     align_corners=False,
    # ).argmax(dim=1)
    
    logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
    ).sigmoid().round()
        
    pred_labels = decode_predictions(logits_tensor.cpu().numpy())

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
writer = SummaryWriter(log_dir='Segformer-mytrainer-oh-bcelossweight-ep100-batch24-augment-splitarea-512-log')
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

