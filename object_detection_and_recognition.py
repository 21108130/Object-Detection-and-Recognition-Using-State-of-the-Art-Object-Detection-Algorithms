# -*- coding: utf-8 -*-
"""Object_Detection_and_Recognition.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15gBM7JHJ7R7pydHHrl9V2U2H-oVEFZoL
"""

# Commented out IPython magic to ensure Python compatibility.
# Install pycocotools
!pip install pycocotools

# Download and extract COCO dataset
!mkdir coco_dataset
# %cd coco_dataset
!mkdir images annotations

# Download train and validation images
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/zips/val2017.zip

# Extract images
!unzip train2017.zip -d images
!unzip val2017.zip -d images


# Download annotations
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract annotations
!unzip annotations_trainval2017.zip -d annotations

# Navigate back to the root directory
# %cd /content

# Now let's load the COCO dataset
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load COCO dataset
coco_train = CocoDetection(root='/content/coco_dataset/images/train2017', annFile='/content/coco_dataset/annotations/annotations/instances_train2017.json', transform=transform)
coco_val = CocoDetection(root='/content/coco_dataset/images/val2017', annFile='/content/coco_dataset/annotations/annotations/instances_val2017.json', transform=transform)

# Create DataLoader
batch_size = 8
train_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(coco_val, batch_size=batch_size, shuffle=False)

import os
import torch
import torchvision
from torchvision import transforms, datasets

# Define the dataset directory
data_dir = '/content/coco_dataset'

# Download and prepare the dataset
dataset = datasets.CocoDetection(root=data_dir, annFile=os.path.join(data_dir, '/content/coco_dataset/annotations/annotations/instances_train2017.json'),
                                 transform=transforms.ToTensor())

# Data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

import torchvision.models.detection as detection

# Load pre-trained Faster R-CNN model
model_faster_rcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_faster_rcnn.eval()

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_faster_rcnn.to(device)

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repository
# %cd yolov5
# %pip install -qr requirements.txt  # Install dependencies

import torch

# Load pre-trained YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_yolo.eval()
model_yolo.to(device)

model_ssd = detection.ssd300_vgg16(pretrained=True)
model_ssd.eval()
model_ssd.to(device)

model_retinanet = detection.retinanet_resnet50_fpn(pretrained=True)
model_retinanet.eval()
model_retinanet.to(device)

import os

# Define the base directory for the dataset
base_dir = '/content/coco_dataset'

# Define subdirectories for images and annotations
dirs = ['train', 'val', 'test']

for d in dirs:
    os.makedirs(os.path.join(base_dir, 'images', d), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'annotations', d), exist_ok=True)

import os
import json
from sklearn.model_selection import train_test_split
import shutil

# Define the base directory for the dataset
base_dir = '/content/coco_dataset'

# Define subdirectories for images and annotations
dirs = ['train', 'val', 'test']

for d in dirs:
    os.makedirs(os.path.join(base_dir, 'images', d), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'annotations', d), exist_ok=True)

# Define paths to the original COCO annotations
coco_annotations_path = os.path.join(base_dir, '/content/coco_dataset/annotations/annotations/instances_train2017.json')

# Load COCO annotations
with open(coco_annotations_path) as f:
    coco_data = json.load(f)

# Get image filenames and corresponding annotation IDs
image_ids = [img['id'] for img in coco_data['images']]
images_info = {img['id']: img for img in coco_data['images']}
annotations_info = {ann['image_id']: [] for ann in coco_data['annotations']}

for ann in coco_data['annotations']:
    annotations_info[ann['image_id']].append(ann)

# Split the dataset
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

# Helper function to save images and annotations
def save_split(ids, split_name):
    images_dir = os.path.join(base_dir, 'images', split_name)
    annotations_dir = os.path.join(base_dir, 'annotations', split_name)

    # Create COCO format annotation file
    split_annotations = {
        'images': [images_info[i] for i in ids],
        'annotations': [ann for i in ids if i in annotations_info for ann in annotations_info[i]],
        'categories': coco_data['categories']
    }

    with open(os.path.join(annotations_dir, f'instances_{split_name}.json'), 'w') as f:
        json.dump(split_annotations, f)

    # Copy images to the split directory
    for i in ids:
        img_info = images_info[i]
        img_filename = img_info['file_name']
        src_path = os.path.join(base_dir, 'images', 'train2017', img_filename)
        dst_path = os.path.join(images_dir, img_filename)
        if os.path.exists(src_path):  # Check if the image file exists before copying
            shutil.copy(src_path, dst_path)

# Save the training, validation, and test splits
save_split(train_ids, 'train')
save_split(val_ids, 'val')
save_split(test_ids, 'test')

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Resize(512, 512),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Example of applying transformations
def augment_image(image, bboxes, class_labels):
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']


image = cv2.imread('/content/coco_dataset/images/test/000000000025.jpg')
bboxes = [[50, 30, 200, 150], [120, 80, 250, 180]]  # Example bounding boxes
class_labels = [1, 2]  # Example class labels

augmented_image, augmented_bboxes, augmented_class_labels = augment_image(image, bboxes, class_labels)

# Convert tensor to numpy array for visualization
augmented_image_np = augmented_image.permute(1, 2, 0).cpu().numpy()

# Display the augmented image using cv2_imshow
cv2_imshow(augmented_image_np)

!pip install torch torchvision torchaudio
!pip install tensorflow
!pip install opencv-python
!pip install albumentations

"""##b. Implementing Faster R-CNN Using PyTorch:"""

import torch
import torchvision.models.detection as detection

# Load pre-trained Faster R-CNN model
model_faster_rcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_faster_rcnn.eval()

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_faster_rcnn.to(device)

import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T

# Load a pre-trained model on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the preprocessing transform
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_path = '/content/coco_dataset/images/test/000000000042.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = transform(image_rgb).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)

print(output)

"""## Implementing YOLO (v5-v9) Using YOLOv5:"""

# Commented out IPython magic to ensure Python compatibility.
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# %pip install -qr requirements.txt  # Install dependencies

import torch

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_yolo.eval()
model_yolo.to(device)

# Commented out IPython magic to ensure Python compatibility.
# Clone the YOLOv5 repository and install requirements
!git clone https://github.com/ultralytics/yolov5
# %cd yolov5
!pip install -r requirements.txt

# Training YOLOv5
!python train.py --img 640 --batch 16 --epochs 50 --data coco.yaml --weights yolov5s.pt

# Inference
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source path_to_image_or_video

"""##Implementing SSD Using PyTorch:"""

# Load pre-trained SSD model
model_ssd = detection.ssd300_vgg16(pretrained=True)
model_ssd.eval()
model_ssd.to(device)

import cv2
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms as T

# Load a pre-trained model on COCO
model = ssd300_vgg16(pretrained=True)
model.eval()

# Define the preprocessing transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((300, 300)),  # SSD300 expects images of size 300x300
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_path = '/content/coco_dataset/images/train/000000000009.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = transform(image_rgb).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)

print(output)

"""##Implementing RetinaNet Using PyTorch:"""

model_retinanet = detection.retinanet_resnet50_fpn(pretrained=True)
model_retinanet.eval()
model_retinanet.to(device)

import cv2
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms as T
from PIL import Image

# Load a pre-trained model on COCO
model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Define the preprocessing transform
transform = T.Compose([
    T.Resize((800, 800)),  # Resize the image to a standard size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_path = '/content/coco_dataset/images/train2017/000000000034.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply the transform
image_pil = Image.fromarray(image_rgb)
image_tensor = transform(image_pil).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)

print(output)

"""##Evaluation
##a. Evaluation Metrics
##Intersection over Union (IoU)
"""

def calculate_iou(box1, box2):
    # Calculate intersection
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate union
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

import cv2
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms as T
from PIL import Image
import time

# Load a pre-trained model on COCO
model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Define the preprocessing transform
transform = T.Compose([
    T.Resize((800, 800)),  # Resize the image to a standard size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_path = '/content/coco_dataset/images/train2017/000000000034.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply the transform
image_pil = Image.fromarray(image_rgb)
image_tensor = transform(image_pil).unsqueeze(0)

# Ensure the image is wrapped in a list
image_list = [image_tensor.squeeze(0)]

# Function to measure inference speed
def measure_inference_speed(model, image, iterations=100):
    # Warm-up run
    model(image)

    # Measure time
    start = time.time()
    for _ in range(iterations):
        model(image)
    end = time.time()

    avg_time = (end - start) / iterations
    return avg_time


avg_inference_time = measure_inference_speed(model, image_list)
print(f'Average inference time: {avg_inference_time} seconds')

import pandas as pd

# Placeholder values for IoU, mAP, and inference times
iou_faster_rcnn = 0.75
iou_yolov5 = 0.70
iou_ssd = 0.65
iou_retinanet = 0.68

map_faster_rcnn = 0.50
map_yolov5 = 0.55
map_ssd = 0.45
map_retinanet = 0.52

time_faster_rcnn = 0.10
time_yolov5 = 0.05
time_ssd = 0.08
time_retinanet = 0.09

# Store the results in a dictionary
results = {
    'Model': ['Faster R-CNN', 'YOLOv5', 'SSD', 'RetinaNet'],
    'IoU': [iou_faster_rcnn, iou_yolov5, iou_ssd, iou_retinanet],
    'mAP': [map_faster_rcnn, map_yolov5, map_ssd, map_retinanet],
    'Inference Time (s)': [time_faster_rcnn, time_yolov5, time_ssd, time_retinanet]
}

# Create a DataFrame from the results
df_results = pd.DataFrame(results)
print(df_results)



"""##images code"""

!pip install pillow
!pip install pycocotools

"""##Faster R-CNN"""

# Install necessary packages
!pip install torch torchvision pycocotools scikit-image

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import random
from skimage import io
import os

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

# Initialize COCO dataset
DATA_PATH = "/content/coco_dataset/images/train2017"
ANN_FILE = os.path.join(DATA_PATH, '/content/coco_dataset/annotations/annotations/instances_train2017.json')
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

# Function to get a random image and its annotations
def get_rand_img(img_ids):
    img_id = random.choice(img_ids)
    img_metadata = coco.loadImgs([img_id])[0]
    img_path = os.path.join(DATA_PATH, '/content/coco_dataset/images/train2017', img_metadata['file_name'])
    img = io.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img, anns

# Function to display ground truth bounding boxes
def display_ground_truth(image, boxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in boxes:
        rect = patches.Rectangle(
            (box['bbox'][0], box['bbox'][1]),
            box['bbox'][2], box['bbox'][3],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Ground Truth")
    plt.show()

# Function to display predicted bounding boxes
def display_predictions(image, outputs):
    fig, ax = plt.subplots()
    ax.imshow(image)
    bboxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    for bbox, score in zip(bboxes, scores):
        if score > 0.5:  # Display predictions with confidence score > 0.5
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=1, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 2, f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    ax.set_title("Predictions")
    plt.show()

# Get a random image and its annotations
img, anns = get_rand_img(img_ids)

# Display the image and its ground truth bounding boxes
plt.imshow(img)
display_ground_truth(img, anns)

# Preprocess the image and move it to the device
img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

# Display the predicted bounding boxes
display_predictions(img, outputs)

"""##Single Shot Detector"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import random
from skimage import io
import os

# Load pre-trained SSD model
model = ssd300_vgg16(pretrained=True)
model = model.to(device)
model.eval()

# Initialize COCO dataset
DATA_PATH = "/content/coco_dataset/images/train2017"
ANN_FILE = os.path.join(DATA_PATH, '/content/coco_dataset/annotations/annotations/instances_train2017.json')
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

# Function to get a random image and its annotations
def get_rand_img(img_ids):
    img_id = random.choice(img_ids)
    img_metadata = coco.loadImgs([img_id])[0]
    img_path = os.path.join(DATA_PATH, '/content/coco_dataset/images/train2017', img_metadata['file_name'])
    img = io.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return img, anns

# Function to display ground truth bounding boxes
def display_ground_truth(image, boxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in boxes:
        rect = patches.Rectangle(
            (box['bbox'][0], box['bbox'][1]),
            box['bbox'][2], box['bbox'][3],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Ground Truth")
    plt.show()

# Function to display predicted bounding boxes
def display_predictions(image, outputs):
    fig, ax = plt.subplots()
    ax.imshow(image)
    bboxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    for bbox, score in zip(bboxes, scores):
        if score > 0.5:  # Display predictions with confidence score > 0.5
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=1, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 2, f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    ax.set_title("Predictions")
    plt.show()

# Get a random image and its annotations
img, anns = get_rand_img(img_ids)

# Display the image and its ground truth bounding boxes
plt.imshow(img)
display_ground_truth(img, anns)

# Preprocess the image and move it to the device
img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

# Display the predicted bounding boxes
display_predictions(img, outputs)

"""##yolov5"""

# Commented out IPython magic to ensure Python compatibility.
# Clone the YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# Install dependencies
!pip install -r requirements.txt

!pip install pycocotools scikit-image

import torch
from pathlib import Path
import random
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Initialize COCO dataset
DATA_PATH = "/content/coco_dataset/images/train2017/"
ANN_FILE = Path("/content/coco_dataset/annotations/annotations/instances_train2017.json")
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO

# Load COCO dataset
DATA_PATH = "/content/coco_dataset/images/train2017/"
ANN_FILE = "/content/coco_dataset/annotations/annotations/instances_train2017.json"
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

def get_rand_img(imgIds):
    img_id = random.choice(imgIds)
    img_metadata = coco.loadImgs([img_id])[0]
    img_path = DATA_PATH + img_metadata['file_name']
    img = Image.open(img_path).convert("RGB")
    annIds = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(annIds)
    return img, anns

def display_ground_truth(image, boxes):
    cpy_img = np.array(image.copy())  # Convert PIL image to numpy array
    fig, ax = plt.subplots()
    ax.imshow(cpy_img)
    for box in boxes:
        rect = patches.Rectangle(
            (int(box['bbox'][0]), int(box['bbox'][1])),
            int(box['bbox'][2]),
            int(box['bbox'][3]),
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Ground Truth")
    plt.show()

def test_model(model, img):
    # Save the image to a temporary file
    img_path = '/content/temp.jpg'
    img.save(img_path)

    # Perform inference
    start_time = time.time()
    results = model(img_path)  # Use the model for inference
    end_time = time.time()

    # Convert the PIL image to a PyTorch tensor
    img_tensor = torchvision.transforms.ToTensor()(img)

    # Move the image tensor to the CPU
    img_array = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format

    # Display the image with bounding boxes
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img_array)  # Display the image

    # Iterate over the detected objects
    for result in results.xyxy[0]:
        if result[4] > 0.5:  # Show boxes with a confidence score above a threshold
            x_min, y_min, x_max, y_max = result[:4].cpu().numpy()
            rect = patches.Rectangle(
                (x_min, y_min),  # (x_min, y_min)
                x_max - x_min,  # width
                y_max - y_min,  # height
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

    inference_time = end_time - start_time
    return ax, inference_time

# Load a random image and its annotations
img, anns = get_rand_img(img_ids)

# Display the ground truth
display_ground_truth(img, anns)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Test the YOLOv5 model on the random image and display the results
ax, t = test_model(model, img)
ax.set_title(f"YOLOv5 \n Time taken: {round(t, 4)} seconds")
plt.show()

"""##retinanet"""

!pip install torch torchvision pycocotools matplotlib

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from pycocotools.coco import COCO

# Load COCO dataset
data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CocoDetection(root='/content/coco_dataset/images/train2017/', annFile='/content/coco_dataset/annotations/annotations/instances_train2017.json', transform=data_transform)

# Define the model
model_retinanet = detection.retinanet_resnet50_fpn(pretrained=True)
model_retinanet.eval()

# Define a function to display ground truth bounding boxes
def display_ground_truth(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for annotation in annotations:
        bbox = annotation['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Define a function to display predicted bounding boxes
def display_predictions(image, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in predictions['boxes']:
        box = box.detach().cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Define a function to get a random image and its annotations
def get_random_image_and_annotations(dataset):
    idx = np.random.randint(0, len(dataset))
    image, annotations = dataset[idx]
    return image, annotations

# Get a random image and its annotations
image, annotations = get_random_image_and_annotations(train_dataset)

# Display ground truth annotations
display_ground_truth(image.permute(1, 2, 0), annotations)

# Preprocess the image
image = image.unsqueeze(0)  # Add batch dimension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
model_retinanet.to(device)

# Run inference
start_time = time.time()
with torch.no_grad():
    predictions = model_retinanet(image)
end_time = time.time()

# Calculate inference time
inference_time = end_time - start_time

# Display predictions
display_predictions(image.squeeze(0).permute(1, 2, 0).cpu(), predictions[0])

# Display inference time
print(f"Inference Time: {inference_time:.4f} seconds")

"""##Evaluation:"""

!pip show yolov5

!pip install yolov5

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import time
import numpy as np

# Function to calculate IoU
def calculate_iou(predictions, target):
    # Ensure predictions and target have at least one bounding box
    if len(predictions['boxes']) == 0 or len(target) == 0:
        return 0.0

    pred_bbox = predictions['boxes'][0].detach().cpu().numpy()
    target_bbox = target[0]['bbox']

    # Calculate the intersection area
    x_min = max(pred_bbox[0], target_bbox[0])
    y_min = max(pred_bbox[1], target_bbox[1])
    x_max = min(pred_bbox[2], target_bbox[0] + target_bbox[2])
    y_max = min(pred_bbox[3], target_bbox[1] + target_bbox[3])

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the union area
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    target_area = target_bbox[2] * target_bbox[3]
    union_area = pred_area + target_area - intersection_area

    # Calculate IoU
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

# Function to calculate mAP (dummy implementation for illustration)
def calculate_mAP(model, dataset):
    # Implement mAP calculation here (code omitted for brevity)
    return 0.0

# Load the test dataset
test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.CocoDetection(root='/content/coco_dataset/images/train2017/', annFile='/content/coco_dataset/annotations/annotations/instances_train2017.json', transform=test_transform)

# Define the models
models = {
    "Faster R-CNN": detection.fasterrcnn_resnet50_fpn(pretrained=True),
    "SSD": detection.ssd300_vgg16(pretrained=True),
    "RetinaNet": detection.retinanet_resnet50_fpn(pretrained=True)
}

# Load YOLOv5 model
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    model.to(device).eval()

    # Run inference and measure inference time
    start_time = time.time()
    total_iou = 0.0
    total_images = 0
    batch_size = 4  # Adjust batch size as needed

    for i in range(0, len(test_dataset), batch_size):
        images, targets = zip(*[test_dataset[j] for j in range(i, min(i + batch_size, len(test_dataset)))])
        images = [image.to(device) for image in images]

        with torch.no_grad():
            predictions = model(images)

        for pred, target in zip(predictions, targets):
            iou = calculate_iou(pred, target)
            total_iou += iou
            total_images += 1

    end_time = time.time()
    inference_time = end_time - start_time

    # Compute mAP
    mAP = calculate_mAP(model, test_dataset)

    # Display results
    print(f"{model_name}:")
    print(f"Mean IoU: {total_iou / total_images}")
    print(f"mAP: {mAP}")
    print(f"Inference time: {inference_time} seconds")
    print("")

# Evaluate YOLOv5
print("Evaluating YOLOv5...")
model_yolov5.to(device).eval()
start_time = time.time()
total_iou = 0.0
total_images = 0

for i in range(0, len(test_dataset), batch_size):
    images, targets = zip(*[test_dataset[j] for j in range(i, min(i + batch_size, len(test_dataset)))])
    images = [image.to(device) for image in images]

    with torch.no_grad():
        predictions = model_yolov5(images)

    for pred, target in zip(predictions, targets):
        iou = calculate_iou(pred, target)
        total_iou += iou
        total_images += 1

end_time = time.time()
inference_time = end_time - start_time

# Display results for YOLOv5
print("YOLOv5:")
print(f"Mean IoU: {total_iou / total_images}")
print(f"Inference time: {inference_time} seconds")