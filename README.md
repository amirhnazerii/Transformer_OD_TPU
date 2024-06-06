# Transformer_OD_TPU
Exploration of TPU Architectures for Optimized Transformer Performance in Image Detection of Drainage Crossings

## Running the Model in Docker
Ensure that the data folder and Transformer_OD_TPU repository are in the same work directory, ```/path/to/work/directory/```.

Navigate to the ```Transformer_OD_TPU``` repository and build the docker image by running:

```docker build -t transformer_od_tpu .```

Then, after building the Docker image, run a jupyter lab server using:

```docker run --gpus all -it -v /path/to/work/directory:/workspace/ -p 8888:8888 transformer_od_tpu```

Then, the program can be run using the following command in the ```Transformer_OD_TPU``` directory:

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output --num_workers 0 --batch_size 8 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth```

This will run the script with default parameters, loading in pretrained object detection weights.

To train from scratch, a binary DETR model can be built by passing ```num_workers 2``` as an argument, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output_scratch --num_workers 0 --batch_size 8 --num_classes 2```
