import warnings
import rasterio 
import numpy as np
from rasterio.windows import Window
import matplotlib.image as image
from PIL import Image
import structlog
from typing import Tuple
import os
import sys
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

log = structlog.get_logger()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_size",
    default=512,
    type=int,
    choices=[512, 1024, 2048, 4096],
    help="Select the image size for cropping",
)
parser.add_argument(
    "--with_test_set",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Specify if the 230816 and 210202 images should be split into train and test sets for image_size 512 and 1024",
)
args = parser.parse_args()


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def adjust_labels_of_tif_files() -> Tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.profiles.Profile]:
    
    with rasterio.open("data/230816_data/breakout.tif") as img1:
        data_breakout = img1.read()
        data_breakout = data_breakout.astype(int)
        data_breakout[data_breakout == 6] = 5
        data_breakout[data_breakout == 0] = 99
    
    with rasterio.open("data/230816_data/breakout_fault_zone.tif") as img2:
        data_breakout_fault_zone = img2.read()
        data_breakout_fault_zone = data_breakout_fault_zone.astype(int)
        data_breakout_fault_zone[data_breakout_fault_zone == 7] = 6 
        data_breakout_fault_zone[data_breakout_fault_zone == 0] = 99

    with rasterio.open("data/230816_data/incipient_breakout.tif") as img3:
        data_incipient_breakout = img3.read()
        data_incipient_breakout = data_incipient_breakout.astype(int)
        data_incipient_breakout[data_incipient_breakout == 4] = 7 
        data_incipient_breakout[data_incipient_breakout == 0] = 99
    
    with rasterio.open("data/230816_data/fault_zone.tif") as img4:
        data_fault_zone = img4.read()
        data_fault_zone = data_fault_zone.astype(int)
        data_fault_zone[data_fault_zone == 5] = 8
        data_fault_zone[data_fault_zone == 0] = 99
    
    with rasterio.open("data/230816_data/tectonic_fault_trace.tif") as img5:
        data_tectonic_fault_trace = img5.read()
        data_tectonic_fault_trace = data_tectonic_fault_trace.astype(int)
        data_tectonic_fault_trace[data_tectonic_fault_trace == 1] = 1
        data_tectonic_fault_trace[data_tectonic_fault_trace == 0] = 99
    
    with rasterio.open("data/230816_data/induced_crack.tif") as img6:
        data_induced_crack = img6.read()
        data_induced_crack = data_induced_crack.astype(int)
        data_induced_crack[data_induced_crack == 2] = 2
        data_induced_crack[data_induced_crack == 0] = 99

    with rasterio.open("data/230816_data/fault_gauge.tif") as img7:
        data_fault_gauge = img7.read()
        data_fault_gauge = data_fault_gauge.astype(int)
        data_fault_gauge[data_fault_gauge == 3] = 3
        data_fault_gauge[data_fault_gauge == 0] = 99
    
    with rasterio.open("data/230816_data/induced_crack_fault_zone.tif") as img8:
        profile = img8.profile
        data_induced_crack_fault_zone = img8.read()
        data_induced_crack_fault_zone = data_induced_crack_fault_zone.astype(int)
        data_induced_crack_fault_zone[data_induced_crack_fault_zone == 8] = 4
        data_induced_crack_fault_zone[data_induced_crack_fault_zone == 0] = 99

    return data_breakout, data_breakout_fault_zone, data_incipient_breakout, data_fault_zone, data_tectonic_fault_trace, data_induced_crack, data_fault_gauge, data_induced_crack_fault_zone, profile


def calculate_class_distribution(all_labels: np.ndarray, num_classes=6) -> None:
    flattened_labels = all_labels.flatten()
    class_counts = np.bincount(flattened_labels, minlength=num_classes)
    total_elements = flattened_labels.size
    class_distribution = (class_counts / total_elements) * 100
    class_names = ['intactwall', 'tectonictrace', 'inducedcrack', 'faultgauge', 'breakout', 'faultzone']
    for class_name, class_dis in zip(class_names, class_distribution):
        log.info(f"The percentage of {class_name} is {round(class_dis, 2)} %")
    return 


def merge_label_to_get_ground_truth() -> Tuple[int, int]:
    data_breakout, data_breakout_fault_zone, data_incipient_breakout, data_fault_zone, data_tectonic_fault_trace, data_induced_crack, data_fault_gauge, data_induced_crack_fault_zone, profile = adjust_labels_of_tif_files()
    all_labels_first = np.minimum(np.minimum(data_breakout, data_breakout_fault_zone), np.minimum(data_incipient_breakout, data_fault_zone))
    all_labels_second = np.minimum(np.minimum(data_tectonic_fault_trace, data_induced_crack), np.minimum(data_fault_gauge, data_induced_crack_fault_zone))
    all_labels = np.minimum(all_labels_first, all_labels_second)
    all_labels[all_labels == 99] = 0
    with rasterio.open('data/230816_data/all_labels.tif', 'w', **profile) as dst:
        dst.write(all_labels[0].astype(rasterio.uint8), 1)


def split_train_test_230816(image_height: int, image_width: int) -> Tuple[int, int, int]:
    percentage_test = 0.3
    percentage_train1 = 0.3
    percentage_train2 = 1 - percentage_test - percentage_train1
    image_height_test = int(image_height*percentage_test)
    image_height_train1 = int(image_height*percentage_train1)
    image_height_train2 = int(image_height*percentage_train2)
    # Create test
    with rasterio.open('data/230816_data/all_labels.tif') as img:
        profile_old = img.profile
        window_test = Window(0, image_height_train1, image_width, image_height_test)
        img_test = img.read(window = window_test)
        profile_test = profile_old.copy()
        profile_test['height'] = image_height_test
        with rasterio.open('data/230816_data/all_labels_test.tif', 'w', **profile_test) as dst:
            dst.write(img_test[0].astype(rasterio.uint8), 1)
    
    with rasterio.open('data/230816_data/230816.tif') as img:
        profile_old = img.profile
        window_test = Window(0, image_height_train1, image_width, image_height_test)
        img_test = img.read(window = window_test)
        profile_test = profile_old
        profile_test['height'] = image_height_test
        directory = 'data/230816_data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.imsave(os.path.join(directory, f'230816_test.jpg'), np.stack((img_test[0],img_test[1],img_test[2]),axis = 2))
    
    # Create top train 
    with rasterio.open('data/230816_data/all_labels.tif') as img:
        profile_old = img.profile
        window_train1 = Window(0, 0, image_width, image_height_train1)
        img_train1 = img.read(window = window_train1)
        profile_train1 = profile_old.copy()
        profile_train1['height'] = image_height_train1
        with rasterio.open('data/230816_data/all_labels_train1.tif', 'w', **profile_train1) as dst:
            dst.write(img_train1[0].astype(rasterio.uint8), 1)
    
    with rasterio.open('data/230816_data/230816.tif') as img:
        profile_old = img.profile
        window_train1 = Window(0, 0, image_width, image_height_train1)
        img_train1 = img.read(window = window_train1)
        profile_train1 = profile_old.copy()
        profile_train1['height'] = image_height_train1
        image.imsave('data/230816_data/230816_train1.jpg', np.stack((img_train1[0],img_train1[1],img_train1[2]),axis = 2))  

    # Create bottom train 
    with rasterio.open('data/230816_data/all_labels.tif') as img:
        profile_old = img.profile
        window_train2 = Window(0, int(image_height*(percentage_test+percentage_train1)), image_width, image_height_train2)
        img_train2 = img.read(window = window_train2)
        profile_train2 = profile_old.copy()
        profile_train2['height'] = image_height_train2
        with rasterio.open('data/230816_data/all_labels_train2.tif', 'w', **profile_train2) as dst:
            dst.write(img_train2[0].astype(rasterio.uint8), 1)
    
    with rasterio.open('data/230816_data/230816.tif') as img:
        profile_old = img.profile
        window_train2 = Window(0, int(image_height*(percentage_test+percentage_train1)), image_width, image_height_train2)
        img_train2 = img.read(window = window_train2)
        profile_train2 = profile_old.copy()
        profile_train2['height'] = image_height_train2
        image.imsave('data/230816_data/230816_train2.jpg', np.stack((img_train2[0],img_train2[1],img_train2[2]),axis = 2))

    return image_height_test, image_height_train1, image_height_train2


def split_train_test_210202(image_height: int, image_width: int) -> Tuple[int, int]:
    # Create test
    percentage_test = 0.2
    image_height_test = int(image_height*percentage_test)
    image_height_train = int(image_height*(1-percentage_test))
    with rasterio.open('data/210202_data/all_labels.tif') as img:
        profile_old = img.profile
        window_test = Window(0, 0, image_width, image_height_test)
        img_test = img.read(window = window_test)
        profile_test = profile_old.copy()
        profile_test['height'] = image_height_test
        with rasterio.open('data/210202_data/all_labels_test.tif', 'w', **profile_test) as dst:
            dst.write(img_test[0].astype(rasterio.uint8), 1)
    
    with rasterio.open(f"data/210202_data/210202.tif") as img: 
        profile_old = img.profile
        window_test = Window(0, 0, image_width, image_height_test)
        img_test = img.read(window = window_test)
        profile_test = profile_old.copy()
        profile_test['height'] = image_height_test
        directory = 'data/210202_data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.imsave(os.path.join(directory, f'210202_test.jpg'), np.stack((img_test[0],img_test[1],img_test[2]),axis = 2))

    # Create train 
    with rasterio.open('data/210202_data/all_labels.tif') as img:
        profile_old = img.profile
        window_train = Window(0, image_height_test, image_width, image_height_train)
        img_train = img.read(window = window_train)
        profile_train = profile_old.copy()
        profile_train['height'] = image_height_train
        with rasterio.open('data/210202_data/all_labels_train.tif', 'w', **profile_train) as dst:
            dst.write(img_train[0].astype(rasterio.uint8), 1)
    
    with rasterio.open(f"data/210202_data/210202.tif") as img:
        profile_old = img.profile
        window_train = Window(0, image_height_test, image_width, image_height_train)
        img_train = img.read(window = window_train)
        profile_train = profile_old.copy()
        profile_train['height'] = image_height_train
        image.imsave('data/210202_data/210202_train.jpg', np.stack((img_train[0],img_train[1],img_train[2]),axis = 2))
    
    return image_height_test, image_height_train


def cropping(
    row_num: int, 
    col_num: int,
    num_files: int,
    n: int,
    width: int,
    height: int,
    path_to_all_labels: str, 
    path_to_image: int,
    path_to_save_annoatation: str,
    path_to_save_image: str,
    multiply_x_by: float,
    multiply_y_by: float,
) -> int:
    with rasterio.open(path_to_all_labels) as img:
        for y in range(row_num): 
            for x in range(col_num):
                if x < col_num-1:
                    col = x*(multiply_x_by)                
                elif x == col_num-1:
                    col = img.width - width                 
                if y < row_num-1:
                    row = y*(multiply_y_by)
                elif y == row_num-1:
                    row = img.height - height
                window = Window(col, row, width, height)
                data_gt = img.read(window = window)
                array = Image.fromarray(data_gt[0].astype(np.uint8),'L')
                if not os.path.exists(path_to_save_annoatation):
                    os.makedirs(path_to_save_annoatation)
                array.save(os.path.join(path_to_save_annoatation, f'annotation_{n}.png'))
                n+=1
        log.info(f"Number of files added to {path_to_save_annoatation}", num_files_added=n-1-num_files)
    n=num_files
    with rasterio.open(path_to_image) as img: 
        for y in range(row_num): 
            for x in range(col_num):
                if x < col_num-1:
                    col = x*(multiply_x_by)                
                elif x == col_num-1:
                    col = img.width - width                 
                if y < row_num-1:
                    row = y*(multiply_y_by)
                elif y == row_num-1:
                    row = img.height - height
                window = Window(col, row, width, height)
                data_gt = img.read(window = window)
                array = Image.fromarray(data_gt[0].astype(np.uint8),'L')
                if not os.path.exists(path_to_save_image):
                    os.makedirs(path_to_save_image)
                image.imsave(os.path.join(path_to_save_image, f'image_{n}.jpg'), np.stack((data_gt[0],data_gt[1],data_gt[2]),axis = 2))
                n+=1
        log.info(f"Number of files added to {path_to_save_image}", num_files_added=n-1-num_files)
    return n - 1 


def cropping_images(
        height: int, 
        width: int, 
        image_height: int, 
        image_width: int, 
        name: str, 
        image_name: str, 
        num_files: int
) -> None:
    log.info(f"Cropping {name} into size {height} x {width}")

    if name == "test":
        train_test = "test"
    elif "train" in name:
        train_test = "train"
    else:
        log.error("Wrong dataset name was given!")

    log.info(f"Number of files currently in {name}", num_files_before=num_files)

    if name != "test": # do overlap cropping
        n = cropping(
            row_num=int(image_height/(height/2)),
            col_num=int(image_width/(width/2)),
            num_files=num_files,
            n=num_files,
            width=width,
            height=height,
            path_to_all_labels=f"data/{image_name}_data/all_labels_{name}.tif",
            path_to_image=f"data/{image_name}_data/{image_name}_{name}.jpg",
            path_to_save_annoatation=f'data/210202_230816/{train_test}_annotation_{height}/',
            path_to_save_image=f'data/210202_230816/{train_test}_image_{height}/',
            multiply_x_by=(width/2),
            multiply_y_by=(height/2),
        )
    else: # no overlap cropping
        n = cropping(
            row_num=int(image_height/height),
            col_num=int(image_width/width),
            num_files=num_files,
            n=num_files,
            width=width,
            height=height,
            path_to_all_labels=f"data/{image_name}_data/all_labels_{name}.tif",
            path_to_image=f"data/{image_name}_data/{image_name}_{name}.jpg",
            path_to_save_annoatation=f'data/210202_230816/{train_test}_annotation_{height}/',
            path_to_save_image=f'data/210202_230816/{train_test}_image_{height}/',
            multiply_x_by=(width),
            multiply_y_by=(height),
        )
    return n


def create_image_jpg(image_name: str) -> None:
    with rasterio.open(f"data/{image_name}_data/{image_name}.tif") as img: 
        img_test = img.read()
        directory = f'data/{image_name}_data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.imsave(os.path.join(directory, f'{image_name}.jpg'), np.stack((img_test[0],img_test[1],img_test[2]),axis = 2))


def cropping_images_all_50_percent_overlap(
        image_name: str, 
        height: int, 
        width: int, 
        image_height: int, 
        image_width: int,
        num_files: int,
) -> int:
    log.info(f"Cropping {image_name} into size {height} x {width}")
    log.info(f"Number of files currently in image_{height} folder", num_files_before=num_files)
    n = cropping(
        row_num=int(image_height/(height/2)),
        col_num=int(image_width/(width/2)),
        num_files=num_files,
        n=num_files,
        width=width,
        height=height,
        path_to_all_labels=f"data/{image_name}_data/all_labels.tif",
        path_to_image=f"data/{image_name}_data/{image_name}.jpg",
        path_to_save_annoatation=f'data/210202_230816/annotation_{height}/',
        path_to_save_image=f'data/210202_230816/image_{height}/',
        multiply_x_by=(width/2),
        multiply_y_by=(height/2),
    )
    return n


def cropping_images_all_20_pixel_overlap(
        height: int, 
        width: int, 
        image_height: int, 
        image_width: int, 
        image_name: str
) -> None:
    log.info(f"Cropping {image_name} into size {height} x {width}")
    _ = cropping(
        row_num=image_height//(width-20)+1,
        col_num=image_width//(width-20)+1,
        num_files=0,
        n=0,
        width=width,
        height=height,
        path_to_all_labels=f"data/{image_name}_data/all_labels.tif",
        path_to_image=f"data/{image_name}_data/{image_name}.jpg",
        path_to_save_annoatation=f'data/201113_data/annotation_{height}/',
        path_to_save_image=f'data/201113_data/image_{height}/',
        multiply_x_by=(width-20),
        multiply_y_by=(height-20),
    )


def cropping_images_no_overlap(
        height: int, 
        width: int, 
        image_height: int, 
        image_width: int, 
        image_name: str
) -> None:
    log.info(f"Cropping {image_name} into size {height} x {width}")
    _ = cropping(
        row_num=int(image_height/height),
        col_num=int(image_width/width),
        num_files=0,
        n=0,
        width=width,
        height=height,
        path_to_all_labels=f"data/{image_name}_data/all_labels.tif",
        path_to_image=f"data/{image_name}_data/{image_name}.jpg",
        path_to_save_annoatation=f'data/{image_name}_data/annotation_{height}_no_overlap/',
        path_to_save_image=f'data/{image_name}_data/image_{height}_no_overlap/',
        multiply_x_by=(width/2),
        multiply_y_by=(height/2),
    )


def get_all_labels(image_name: str) -> Tuple[np.ndarray, int, int]:
    with rasterio.open(f'data/{image_name}_data/all_labels.tif') as img:
        log.info(f"Image shape of {image_name}: ", shape=img.shape)
        image_width = img.width
        image_height = img.height
        all_labels = img.read()
    all_labels = all_labels.astype(int)
    return all_labels, image_width, image_height


def get_210202() -> Tuple[int, int]:
    all_labels, image_width, image_height = get_all_labels(image_name="210202")
    log.info("Class distribution of image 210202")
    calculate_class_distribution(all_labels)

    if (args.image_size==512 or args.image_size==1024) and args.with_test_set:
        image_height_test, image_height_train = split_train_test_210202(image_height=image_height, image_width=image_width)
        num_test = cropping_images(
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height_test, 
            image_width=image_width, 
            name="test", 
            image_name="210202",
            num_files=0,
        )
        num_train = cropping_images(
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height_train, 
            image_width=image_width, 
            name="train", 
            image_name="210202",
            num_files=0,
        )
    else: 
        create_image_jpg(image_name="210202")
        num_train = cropping_images_all_50_percent_overlap(
            image_name="210202",
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height, 
            image_width=image_width, 
            num_files=0,
        )
        num_test = 0
    return num_test, num_train


def get_230816(num_test: int, num_train: int) -> None:
    merge_label_to_get_ground_truth()
    all_labels, image_width, image_height = get_all_labels(image_name="230816")
    log.info("Class distribution of image 230816")
    calculate_class_distribution(all_labels)

    if (args.image_size==512 or args.image_size==1024) and args.with_test_set:
        image_height_test, image_height_train1, image_height_train2 = split_train_test_230816(image_height=image_height, image_width=image_width)
        _ = cropping_images(
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height_test, 
            image_width=image_width, 
            name="test", 
            image_name="230816",
            num_files=num_test,
        )
        num_train1 = cropping_images(
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height_train1, 
            image_width=image_width, 
            name="train1", 
            image_name="230816",
            num_files=num_train,
        )
        _ = cropping_images(
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height_train2, 
            image_width=image_width, 
            name="train2", 
            image_name="230816",
            num_files=num_train1,
        )
    else: 
        create_image_jpg(image_name="230816")
        _ = cropping_images_all_50_percent_overlap(
            image_name="230816",
            height=args.image_size, 
            width=args.image_size, 
            image_height=image_height, 
            image_width=image_width, 
            num_files=num_train,
        )


def get_201113() -> None:
    all_labels, image_width, image_height = get_all_labels(image_name="201113")
    log.info("Class distribution of image 201113")
    calculate_class_distribution(all_labels)
    cropping_images_all_20_pixel_overlap(
        height=args.image_size, 
        width=args.image_size, 
        image_height=image_height, 
        image_width=image_width, 
        image_name="201113",
    )
    cropping_images_no_overlap(
        height=args.image_size, 
        width=args.image_size, 
        image_height=image_height, 
        image_width=image_width, 
        image_name="201113",
    )


def check_image_sizes(directory) -> None:
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))]
    for file in files:
        with Image.open(os.path.join(directory, file)) as img:
            size = img.size
            height = size[0]
            width = size[1]
            assert height == args.image_size
            assert width == args.image_size
    log.info(f"All files in the directory {directory} have size {args.image_size} x {args.image_size}.")


def check_that_image_annotation_have_same_num_files(directory: str) -> None:
    annotation_dir = directory + f'annotation_{args.image_size}'
    image_dir = directory + f'image_{args.image_size}'
    files_annotation = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]
    files_image = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    assert len(files_annotation) == len(files_image)
    log.info(f"The {annotation_dir} and the {image_dir} directory contain the same number of files.")


if __name__ == "__main__":
    num_test, num_train = get_210202()
    num_test = 0
    num_train = 0
    get_230816(num_test=num_test, num_train=num_train)
    get_201113()

    # Tests
    if (args.image_size==512 or args.image_size==1024) and args.with_test_set:
        for dir in ["data/210202_230816/test_", "data/210202_230816/train_", "data/201113_data/"]:
            check_that_image_annotation_have_same_num_files(directory=dir)
        dirs = [
            f"data/210202_230816/test_annotation_{args.image_size}",
            f"data/210202_230816/test_image_{args.image_size}",
            f"data/210202_230816/train_annotation_{args.image_size}",
            f"data/210202_230816/train_image_{args.image_size}",
            f"data/201113_data/annotation_{args.image_size}",
            f"data/201113_data/image_{args.image_size}"
        ]
    else: 
        for dir in ["data/210202_230816/", "data/201113_data/"]:
            check_that_image_annotation_have_same_num_files(directory=dir)
        dirs = [
            f"data/210202_230816/annotation_{args.image_size}",
            f"data/210202_230816/image_{args.image_size}",
            f"data/201113_data/annotation_{args.image_size}",
            f"data/201113_data/image_{args.image_size}"
        ]
    for dir in dirs: 
        check_image_sizes(directory=dir)
