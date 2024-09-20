import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import rasterio 
import cv2
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import structlog
from typing import Tuple, Dict
import seaborn as sns
log = structlog.get_logger()

from utils import UnetModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_size",
    default=512,
    type=int,
    choices=[512, 1024, 2048, 4096],
    help="Select the image size, e.g. 512, 1024",
)
parser.add_argument(
    "--model_name",
    default="unet_ep500_focalloss_1309",
    type=str,
    help="Give the name of the model, e.g. unet_ep500_focalloss_1309",
)
parser.add_argument(
    "--get_uncertainty",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Specify if the uncertainty should be calculated",
)
parser.add_argument(
    "--get_predictmap",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Specify if the prediction map should be calculated. If get_uncertainty is True, then get_predictmap will not be executed.",
)
args = parser.parse_args()


def load_checkpoint(model_path: str) -> UnetModel:
    model = UnetModel.load_from_checkpoint(
        model_path, 
        train_loader=None,
        device=torch.device('cpu')
    )
    model.eval()
    return model


def process_image_unet(image_path) -> torch.Tensor:
    image_test = cv2.imread(image_path)
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    image_test = image_test.transpose(2,0,1)
    image_test = np.expand_dims(image_test, axis=0)
    image_test = torch.from_numpy(image_test)
    return (image_test)


def calculate_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-10), axis=1) 


def get_full_test_image(model: UnetModel, directory: str) -> Tuple[int, int]:
    with rasterio.open(os.path.join(directory, 'all_labels.tif')) as img:
        image_width = img.width
        image_height = img.height
        profile = img.profile

    allresult = np.zeros((1, image_height, image_width))
    width = args.image_size
    height = args.image_size
    image_directory = os.path.join(directory, f'image_{args.image_size}')
    num_image = len([f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))])
    col_num = image_width//(width-20)+1
    row_num = image_height//(width-20)+1 

    for i in tqdm(range(num_image)):
        inputs = process_image_unet(os.path.join(directory, f'image_{args.image_size}/image_{i}.jpg'))
        logits = model(inputs) # shape (batch_size, num_labels, height/4, width/4)
        if args.get_uncertainty:
            probability = nn.functional.softmax(logits, dim=1) * 100
            p_array = probability.detach().numpy()
            entropy_map = calculate_entropy(p_array)
            pred_arr = entropy_map[0]
        elif args.get_predictmap:
            pred_seg = logits.argmax(dim=1)[0]
            pred_arr = np.array(pred_seg)
        
        row_idx = i//col_num
        col_idx = i%col_num
        if col_idx < col_num-1:
            col = col_idx*(width-20)                
        elif col_idx == col_num-1:
            col = image_width - width                 
        if row_idx < row_num-1:
            row = row_idx*(height-20)
        elif row_idx == row_num-1:
            row = image_height - height

        if col_idx==0 or col_idx==(col_num-1) or row_idx==0 or row_idx==(row_num-1):
            allresult[:, int(row):int(row)+height, int(col):int(col)+width] = pred_arr
        else:
            allresult[:, int(row)+10:int(row)+height-10, int(col) +10:int(col)+width-10] = pred_arr[10:-10,10:-10]
    assert allresult.shape[0] == 1
    assert allresult.shape[1] == image_height
    assert allresult.shape[2] == image_width

    if args.get_uncertainty:
        allresult = allresult*100
        with rasterio.open(f'data/201113_data/prediction_of_{args.model_name}_with_uncertainty.tif', 'w', **profile) as dst:
            dst.write(allresult[0].astype(rasterio.uint8), 1)
    else: 
        with rasterio.open(f'data/201113_data/prediction_of_{args.model_name}.tif', 'w', **profile) as dst:
            dst.write(allresult[0].astype(rasterio.uint8), 1)
    return image_width, image_height


def compare_groundtruth_with_prediction(groundtruth_path: str, pred_result_path: str, classes: Dict[int, str]):
    classes = list(classes.keys())
    cm = np.zeros((6,6))
    with rasterio.open(groundtruth_path) as gtimg:
        groundtruth = gtimg.read()
    with rasterio.open(pred_result_path) as predimg:
        prediction = predimg.read()
    ground_truth = groundtruth.flatten()
    predictions = prediction.flatten()  
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=classes)
    cm = cm+conf_matrix
    print("type(cm): ", type(cm))
    return cm


def calc_performance(cm: np.ndarray, classes: Dict[int, str]) -> None:
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    log.info(f'Overall Accuracy: {accuracy:.4f}')

    iou_per_class = []
    for i in range(6):
        intersection = cm[i, i]
        union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - intersection
        iou = intersection / union if union != 0 else 0
        iou_per_class.append(iou)
        log.info(f'IoU for class {classes[i]}: {iou:.4f}')

    mean_iou = np.mean(iou_per_class)
    log.info(f'Mean IoU: {mean_iou:.4f}')


def plot_confucion_matrix(cm: np.ndarray, classes: Dict[int, str]) -> None:
    classes = list(classes.values())
    conf_matrix_norm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix_norm, annot=True,fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    save_name = f'figures/prediction_of_model_{args.model_name}_on_201113.png'
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    print("Model path: ", f'unet_checkpoint/{args.model_name}.ckpt')
    model = load_checkpoint(model_path=f'unet_checkpoint/{args.model_name}.ckpt')
    log.info("Checkpoint was loaded")
    image_width, image_height = get_full_test_image(
        model=model, 
        directory="data/201113_data",
    )
    groundtruth_path = "data/201113_data//all_labels.tif"
    pred_result_path = f"data/201113_data/prediction_of_{args.model_name}.tif"
    classes = {
        0: 'intactwall', 
        1: 'tectonictrace', 
        2: 'inducedcrack', 
        3: 'faultgauge', 
        4: 'breakout', 
        5: 'faultzone',
    }
    cm = compare_groundtruth_with_prediction(
        groundtruth_path=groundtruth_path, 
        pred_result_path=pred_result_path, 
        classes=classes,
    )
    calc_performance(cm=cm, classes=classes)
    plot_confucion_matrix(cm=cm, classes=classes)
