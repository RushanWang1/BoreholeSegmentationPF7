import torch
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import structlog
log = structlog.get_logger()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='sum', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        device = inputs.device
        logpt = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-logpt)
        
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            at = alpha[targets]
            logpt = logpt * at
        
        focal_loss = (1 - pt) ** self.gamma * logpt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class UnetModel(pl.LightningModule):
    def __init__(
            self, 
            train_loader=None,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            arch = "Unet", 
            loss_function="FocalLoss",
            encoder_name="resnet34",
            in_channels = 3, # for RGB images
            out_classes = 6, 
            initial_learning_rate=2e-4,
            **kwargs
    ):
        super().__init__()
        self.train_loader = train_loader
        self.initial_learning_rate = initial_learning_rate
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            in_channels=in_channels, 
            classes=out_classes, 
            **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if self.train_loader:
            class_weights = self.compute_class_weights(out_classes)
            class_weights_tensor = class_weights.float().clone().detach().to(device) 
        else: 
            class_weights_tensor = None

        if loss_function == "FocalLoss":
            self.loss_fn = FocalLoss(
                alpha=class_weights_tensor, 
                gamma=2, 
                reduction='mean' # Changed to mean: the average of all element-wise losses is returned, so the loss scales independently of the batch size. When using 'sum' the total loss ncreases proportionally with the number of pixels.
            ).to(device) 
        elif loss_function == "CrossEntropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor
            ).to(device)
        else: 
            log.error("Incorrect loss function was chosen, please select either 'FocalLoss', or 'CrossEntropy'")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.t_max = None
        self.losses = []
        self.validation_losses = []
    
    def compute_class_weights(self, out_classes):
        class_counts = torch.zeros(out_classes, dtype=torch.float32)

        for batch in self.train_loader:
            _, labels = batch 
            for class_idx in range(out_classes):
                class_counts[class_idx] += (labels == class_idx).sum().item()

        class_weights = 1.0 / class_counts
        class_weights[class_counts == 0] = 0
        class_weights /= class_weights.sum() 
        return class_weights
    
    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch):
        image, mask = batch
        assert image.ndim == 4 # Shape of the image should be (batch_size, num_channels, height, width)
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0 # Check that image dimensions are divisible by 32
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        gt_label = mask.argmax(axis=1) 
        loss = self.loss_fn(logits_mask, gt_label)

        tp, fp, fn, tn = smp.metrics.get_stats(
            logits_mask.argmax(axis = 1).long(), 
            gt_label.long(), 
            num_classes = 6, 
            mode="multiclass"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        mean_iou = dataset_iou.mean().item() 
        metrics = {
            f"{stage}_mean_iou": round(mean_iou*100, 2),
        }
        self.log_dict(metrics, prog_bar=True)
    
    def compute_all_metrics(self, outputs):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item() 
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        mean_iou = dataset_iou.mean().item() 
        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").mean(dim=0).tolist()
        
        metrics = {
            "test_per_image_iou": round(per_image_iou * 100, 2),
            "test_mean_iou": round(mean_iou * 100, 2),
            "test_iou_intactwall": round(iou_per_class[0] * 100, 2),
            "test_iou_tectonictrace": round(iou_per_class[1] * 100, 2),
            "test_iou_inducedcrack": round(iou_per_class[2] * 100, 2),
            "test_iou_faultgauge": round(iou_per_class[3] * 100, 2),
            "test_iou_breakout": round(iou_per_class[4] * 100, 2),
            "test_iou_faultzone": round(iou_per_class[5] * 100, 2),
        }
        return metrics

    def training_step(self, batch):
        train_loss_info = self.shared_step(batch)
        self.training_step_outputs.append(train_loss_info)
        self.losses.append(train_loss_info["loss"].item())
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, stage='train')
        train_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        self.training_step_outputs.clear()
        return 

    def validation_step(self, batch):
        valid_loss_info = self.shared_step(batch)
        self.validation_step_outputs.append(valid_loss_info)
        self.validation_losses.append(valid_loss_info["loss"].item())
        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, stage="val")
        val_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()
        return 
    
    def test_step(self, batch):
        test_loss_info = self.shared_step(batch)
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        test_metrics = self.compute_all_metrics(self.test_step_outputs)
        self.log_dict(test_metrics)
        self.test_step_outputs.clear()
        return test_metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.t_max, 
            eta_min=1e-5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    