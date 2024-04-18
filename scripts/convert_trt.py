#!/usr/bin/env python3

# from https://docs.ultralytics.com/integrations/tensorrt/
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/root/sim_ws/src/lab8_pkg/yolo/best.pt')

# Export the model to TensorRT format
model.export(format='engine')  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO('yolov8n.engine')

# Run inference
results = tensorrt_model('/root/sim_ws/src/lab8_pkg/yolo/1431.jpg')

# Save the resulting image
results.save('/root/sim_ws/src/lab8_pkg/yolo/yolo_result.jpg')

print("Inference result saved as result.jpg.")
