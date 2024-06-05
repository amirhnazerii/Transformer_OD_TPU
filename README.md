# Transformer_OD_TPU
Exploration of TPU Architectures for Optimized Transformer Performance in Image Detection of Drainage Crossings

## Running the Model in Docker
Ensure that the data folder and Transformer_OD_TPU repository are in the same work directory.

Then, after building the Docker image using the Dockerfile, run a jupyter lab server using:

```docker run --gpus all -it -v /path/to/work/directory:/workspace/ -p 8888:8888 transformer_od_tpu```

Then, the program can be run using the following command in the ```Transformer_OD_TPU``` directory:

```torchrun main.py --coco_path /path/to/data --output_dir /path/to/output --num_workers 0 --batch_size 8 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth```

This will run the script with default parameters, loading in pretrained object detection weights.
