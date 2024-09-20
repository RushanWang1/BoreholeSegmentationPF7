
import os
import sys
import cv2
import torch
import numpy as np
from typing import Tuple, Dict, List
import ssl
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset as BaseDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import argparse
import structlog
log = structlog.get_logger()

from utils import UnetModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_workers",
    default=8,
    type=int,
    help="Select the number of workers",
)
parser.add_argument(
    "--epochs",
    default=1,
    type=int,
    help="Select the number of epochs",
)
parser.add_argument(
    "--encoder_name",
    default="resnet34",
    type=str,
    choices=[
        "resnet18", # Num params 11M
        "resnet34", # Num params 21M
        "resnet50", # Num params 32M
        "resnet101", # Num params 42M
        "resnet152", # Num params 58M
        "resnext50_32x4d", # Num params 22M
        "resnext101_32x8d", # Num params 86M
        "efficientnet-b0", # Num params 4M
        "efficientnet-b1", # Num params 6M
        "efficientnet-b2", # Num params 7M
        "efficientnet-b3", # Num params 10M
        "efficientnet-b4", # Num params 17M
        "efficientnet-b5", # Num params 28M
        "efficientnet-b6", # Num params 40M
        "efficientnet-b7", # Num params 63M
        "vgg13", # Num params 9M
        "vgg16", # Num params 14M
        "vgg19", # Num params 20M
        "densenet121", # Num params 6M
        "densenet169", # Num params 12M
        "densenet201", # Num params 18M
        "densenet161", # Num params 26M
    ],
    help="Select the encoder, e.g. 'resnet18' or 'resnet34'",
)
parser.add_argument(
    "--val_split",
    default=0.05,
    type=float,
    help="Percentage of the training dataset that should be used validation",
)
parser.add_argument(
    "--random_seed",
    default=42,
    type=int,
    help="Select the random seed for reproducability",
)
parser.add_argument(
    "--image_size",
    default=512,
    type=int,
    choices=[512, 1024, 2048, 4096],
    help="Select the image size, e.g. 512, 1024",
)
parser.add_argument(
    "--augmented",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Decide if you want to use the augmented data for training or not",
)
parser.add_argument(
    "--initial_learning_rate",
    default=2e-4,
    type=float,
    help="Set the initial leanring rate for the scheduler",
)
parser.add_argument(
    "--loss_function",
    default="FocalLoss",
    type=str,
    choices=["FocalLoss", "CrossEntropy"],
    help="Decide which loss function should be used",
)
parser.add_argument(
    "--batch_size_train_val",
    default=16,
    type=int,
    help="Select the batch size of the training and validation set",
)
parser.add_argument(
    "--batch_size_test",
    default=8,
    type=int,
    help="Select the batch size of the test set",
)
args = parser.parse_args()


def get_train_test_dir():
    x_train_dir = f'data/210202_230816/image_{args.image_size}'
    y_train_dir = f'data/210202_230816/annotation_{args.image_size}'
    x_test_dir = f'data/201113_data/image_{args.image_size}_no_overlap'
    y_test_dir = f'data/201113_data/annotation_{args.image_size}_no_overlap'
    x_train_augmented_dir = f'data/210202_230816/augmented/image_{args.image_size}'
    y_train_augmented_dir = f'data/210202_230816/augmented/annotation_{args.image_size}'
    return x_train_dir, y_train_dir, x_test_dir, y_test_dir, x_train_augmented_dir, y_train_augmented_dir


def get_training_augmentation(aug: A.augmentations) -> A.core.composition.Compose:
    train_transform = [
        aug
    ]
    return A.Compose(transforms=train_transform)


def apply_augmentation_and_save() -> None:
    images_dir, masks_dir, _, _, save_images_dir, save_masks_dir = get_train_test_dir()
    num_classes = 6
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
    if not os.path.exists(save_masks_dir):
        os.makedirs(save_masks_dir)

    transformations = {
        "none": None, 
        "h_flip": A.HorizontalFlip(p=1),
        "v_flip": A.VerticalFlip(p=1),
        "rotate": A.Rotate(limit=30, p=1),
        "crop": A.RandomCrop(width=args.image_size, height=args.image_size, p=1),
        "color": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        "contrast": A.CLAHE(p=1),
        "blur": A.GaussianBlur(blur_limit=(3, 7), p=1),
    }

    for name, aug in transformations.items():
        log.info(f"Carrying out the {name} augmentation")
        if name != "none":
            augmentation = get_training_augmentation(aug)
        image_fps = [os.path.join(images_dir, file) for file in sorted(os.listdir(images_dir))]
        mask_fps = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir))]

        for img_fp, mask_fp in tqdm(zip(image_fps, mask_fps), total=len(image_fps)):
            image = cv2.imread(img_fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            mask = cv2.imread(mask_fp, 0)
            masks = [(mask == v) for v in range(num_classes)]  
            mask = np.stack(masks, axis=-1).astype(np.float32)

            if name != "none":
                augmented = augmentation(image=image, mask=mask)
                aug_image, aug_mask = augmented['image'], augmented['mask']
            else: 
                aug_image = image
                aug_mask = mask

            filename = os.path.basename(img_fp)
            save_image_path = os.path.join(save_images_dir, f'{os.path.splitext(filename)[0]}_{name}.jpg')
            cv2.imwrite(save_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            save_mask_path = os.path.join(save_masks_dir, f'{os.path.splitext(filename)[0]}_{name}.png')
            single_channel_mask = np.argmax(aug_mask, axis=-1).astype(np.uint8)
            cv2.imwrite(save_mask_path, single_channel_mask)


class Dataset(BaseDataset):
    CLASSES = ['intactwall', 'tectonictrace', 'inducedcrack', 'faultgauge', 'breakout', 'faultzone']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
    ):
        self.ids = os.listdir(images_dir)
        self.ids_annotation = os.listdir(masks_dir)  
        if "augmented" in images_dir:
            self.ids.sort(key=lambda x: (int(x.split('_')[1]), x.split('_')[2]))
            self.ids_annotation.sort(key=lambda x: (int(x.split('_')[1]), x.split('_')[2]))
        else: 
            self.ids.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
            self.ids_annotation.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_annotation]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            log.error(f"Image not found at index{i}: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask = cv2.imread(self.masks_fps[i], 0)
        if mask is None:
            log.error(f"Mask not found at index {i}: {self.masks_fps[i]}")
        masks = [(mask == v) for v in self.class_values] 
        mask = np.stack(masks, axis=-1).astype(np.float32) 
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)


def calculate_class_distribution(dataset: torch.utils.data.dataset.Subset, num_classes=6) -> List[float]:
    total_pixels = 0
    class_counts = Counter()
    for _, label in dataset:
        if label.ndim == 3:
            label = np.argmax(label, axis=0)
        flattened_labels = label.flatten()
        assert len(flattened_labels) == args.image_size * args.image_size
        class_counts.update(flattened_labels)
        total_pixels += flattened_labels.size
    class_distribution = [(class_counts[i]/total_pixels*100) for i in range(num_classes)]
    return class_distribution

def get_train_test_datasets(num_workers: int) -> Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]:
    x_train_dir, y_train_dir, x_test_dir, y_test_dir, x_train_augmented_dir, y_train_augmented_dir = get_train_test_dir()
    
    classes = ['intactwall', 'tectonictrace', 'inducedcrack', 'faultgauge', 'breakout', 'faultzone']
    
    if args.augmented: 
        train_dataset = Dataset(
            x_train_augmented_dir, 
            y_train_augmented_dir, 
            classes=classes
        )
        train_dataset_size_old = len(train_dataset)
        image_groups = defaultdict(list) # Make sure that augmentations of the same image are all in either the train or val set
        for idx, image_id in enumerate(train_dataset.ids):
            original_image_id = '_'.join(image_id.split('_')[:2]) 
            image_groups[original_image_id].append(idx)
        grouped_indices = list(image_groups.values())
        total_groups = len(grouped_indices)
        val_size = int(total_groups * args.val_split) 
        train_size = total_groups - val_size
        torch.manual_seed(args.random_seed)
        permuted_indices = torch.randperm(total_groups).tolist()
        train_group_indices = [grouped_indices[i] for i in permuted_indices[:train_size]]
        val_group_indices = [grouped_indices[i] for i in permuted_indices[train_size:]]
        train_indices = [idx for group in train_group_indices for idx in group]
        val_indices = [idx for group in val_group_indices for idx in group]
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
        assert train_dataset_size_old == len(train_dataset) + len(val_dataset)

    else: 
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            classes=classes
        )
        total_train_size = len(train_dataset)
        val_size = int(total_train_size * args.val_split)
        train_size = total_train_size - val_size
        torch.manual_seed(args.random_seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    log.info("Trainingset class distribution")
    train_class_distribution = calculate_class_distribution(train_dataset)
    log.info("{}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%".format(
            classes[0], train_class_distribution[0],
            classes[1], train_class_distribution[1],
            classes[2], train_class_distribution[2],
            classes[3], train_class_distribution[3],
            classes[4], train_class_distribution[4],
            classes[5], train_class_distribution[5])
    )
    log.info("Validationset class distribution")
    val_class_distribution = calculate_class_distribution(val_dataset)
    log.info("{}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%".format(
            classes[0], val_class_distribution[0],
            classes[1], val_class_distribution[1],
            classes[2], val_class_distribution[2],
            classes[3], val_class_distribution[3],
            classes[4], val_class_distribution[4],
            classes[5], val_class_distribution[5])
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size_train_val,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size_train_val, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True,
    )
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=classes,
    )
    log.info("Testset class distribution")
    test_class_distribution = calculate_class_distribution(test_dataset)
    log.info("{}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%".format(
            classes[0], test_class_distribution[0],
            classes[1], test_class_distribution[1],
            classes[2], test_class_distribution[2],
            classes[3], test_class_distribution[3],
            classes[4], test_class_distribution[4],
            classes[5], test_class_distribution[5])
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


def plot_loss(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Steps')
    plt.legend()
    plt.grid(True)
    save_name = f'figures/unet_train_loss_plot_ep{args.epochs}_encoder_{args.encoder_name}_{args.image_size}_{args.loss_function}'
    if args.augmented:
        save_name = save_name + '_augmented'
    plt.savefig(f"{save_name}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 0.2) 
    plt.title('Validation Loss vs. Steps')
    plt.legend()
    plt.grid(True)
    save_name = f'figures/unet_val_loss_plot_ep{args.epochs}_encoder_{args.encoder_name}_{args.image_size}_{args.loss_function}'
    if args.augmented:
        save_name = save_name + '_augmented'
    plt.savefig(f"{save_name}.png")
    plt.close()


def train_unet(train_loader: torch.utils.data.dataloader.DataLoader, val_loader: torch.utils.data.dataloader.DataLoader):
    t_max = args.epochs * len(train_loader)
    print("t_max: ", t_max)

    model = UnetModel(
        train_loader=train_loader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        arch="Unet", 
        loss_function=args.loss_function,
        encoder_name=args.encoder_name,
        in_channels=3, 
        out_classes=6,
        initial_learning_rate=args.initial_learning_rate,
    )
    
    model.t_max = t_max
    log.info("Model was created")

    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        log_every_n_steps=1, 
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.fit(
        model.to(device), 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
    )
    model_path = f'unet_checkpoint/unet_ep{args.epochs}_encoder_{args.encoder_name}_{args.image_size}_{args.loss_function}'
    if args.augmented:
        model_path = f'{model_path}_augmented'
    trainer.save_checkpoint(f"{model_path}.ckpt") 

    plot_loss(train_losses=model.losses, val_losses=model.validation_losses)
    return model, trainer


def test_model(model: UnetModel, test_loader: torch.utils.data.dataloader.DataLoader, trainer) -> None:
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False) 
    for name, metric in test_metrics[0].items():
        log.info("{}: {:.2f}%".format(name, metric)) 


if __name__ == "__main__":
    if args.augmented and not os.path.exists("data/210202_230816/augmented"):
        apply_augmentation_and_save()
    train_loader, val_loader, test_loader = get_train_test_datasets(num_workers=args.num_workers)
    log.info("Data was loaded, now training the model")
    model, trainer = train_unet(train_loader, val_loader)
    log.info("Model was trained, now testing the model")
    mean_iou = test_model(model=model, test_loader=test_loader, trainer=trainer)

