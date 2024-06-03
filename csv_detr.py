import torch
import csv
from transformers import DetrModel, DetrConfig, DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Input image dimensions
input_height = 224
input_width = 224

# output dimensions after a convolutional layer
def conv_output_dim(input_size, kernel_size, stride, padding=0, dilation=1):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

# Extract layer information
def extract_layer_info(layer, current_height, current_width):
    if not isinstance(layer, torch.nn.Conv2d):
        return None, current_height, current_width

    layer_info = {
        "Layer name": layer.__class__.__name__,
        "IFMAP Height": current_height,
        "IFMAP Width": current_width,
        "Filter Height": layer.kernel_size[0],
        "Filter Width": layer.kernel_size[1],
        "Channels": layer.in_channels,
        "Num Filter": layer.out_channels,
        "Strides": layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    }

    current_height = conv_output_dim(current_height, layer.kernel_size[0], layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride, layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding)
    current_width = conv_output_dim(current_width, layer.kernel_size[1], layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride, layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding)
    
    return layer_info, current_height, current_width

# Extract layer information from the model
layer_info_list = []
current_height, current_width = input_height, input_width

for name, layer in model.named_modules():
    if len(list(layer.children())) == 0:  # Skip modules with children
        layer_info, current_height, current_width = extract_layer_info(layer, current_height, current_width)
        if layer_info:  # Only add if it's a convolutional layer
            layer_info["Layer name"] = name  # Update with actual layer name
            layer_info_list.append(layer_info)

# Define the CSV header
header = ["Layer name", "IFMAP Height", "IFMAP Width", "Filter Height", "Filter Width", "Channels", "Num Filter", "Strides"]

# Write to CSV with an extra comma at the end of each line
with open('detr_conv_layer_info.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for layer_info in layer_info_list:
        row = {key: str(value) for key, value in layer_info.items()}
        row_string = ",".join(row.values()) + ","
        csvfile.write(row_string + "\n")

