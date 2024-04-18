#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from torchvision import transforms
from your_model_module import F110_YOLO  # Make sure to import your model class


def save_image_with_boxes(image, bboxs, output_path):
    # Convert image to RGB
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Create figure and axis
    fig, ax = plt.subplots(1)

    # Plot image
    ax.imshow(image)

    # Define edge color for bounding boxes
    edgecolor = [1, 0, 0]  # Red color

    # Plot bounding boxes
    for bbox in bboxs:
        rect = patches.Rectangle((bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2),
                                 bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)

    # Save the figure
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to release memory


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = F110_YOLO()
model_save_name = 'best_350.pt'
path = f"/root/sim_ws/src/lab8_pkg/{model_save_name}"
model.load_state_dict(torch.load(path))
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other necessary transformations such as normalization
])

# Load and preprocess the input image
image_path = "/root/sim_ws/src/lab8_pkg/dataset/0.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# Add batch dimension and move to device
image = transform(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    model.eval()
    output_bboxs = model(image)
    print(output_bboxs)

# Process the output as needed
# For example, you might extract bounding box coordinates, class predictions, etc.
# For demonstration purposes, let's assume 'output_bboxs' is a list of bounding boxes

# Example bounding box coordinates
# output_bboxs = [[100, 100, 50, 50], [200, 200, 30, 30]]
# Path to save the output image with bounding boxes
output_image_path = "/root/sim_ws/src/lab8_pkg/output_with_boxes.jpg"

# Save the resulting image with bounding boxes
save_image_with_boxes(cv2.imread(image_path), output_bboxs, output_image_path)
