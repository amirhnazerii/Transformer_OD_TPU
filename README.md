# Transformer_OD_TPU
Exploration of TPU Architectures for Optimized Transformer Performance in Image Detection of Drainage Crossings

## Running the Model in Docker
Ensure that the data folder and Transformer_OD_TPU repository are in the same work directory, ```/path/to/work/directory/```.

Navigate to the ```Transformer_OD_TPU``` repository and build the docker image by running:

```docker build -t transformer_od_tpu .```

Then, after building the Docker image, run a jupyter lab server using:

```docker run --gpus all -it -v /path/to/work/directory:/workspace/ -p 8888:8888 transformer_od_tpu```

Then, the program can be run using the following command in the ```Transformer_OD_TPU``` directory:

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output --num_workers 0 --batch_size 8 --crop 400 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth```

This will run the script with default parameters, loading in pretrained object detection weights.

To train from scratch, a binary DETR model can be built by passing ```num_classes 2``` as an argument, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output_scratch --num_workers 0 --batch_size 8 --crop 400 --num_classes 2```

## Dataset

In order to test model transferrability to an unseen watershed, we first partition the data into the initial dataset and the transfer dataset.

The initial dataset, which includes watersheds from NE, CA, and IL, is randomly paritioned into training (70%), validation (20%), and testing (10%) sets.

The transfer dataset, which includes data from the ND watershed, is used in its entirety to test model transferrability to unseen HRDEM data.

The initial and transfer dataset are normalized according to their corresponding dataset mean and standard deviation to simulate the model's use in inference. It is expected that for a given inference dataset, the user will first find the mean and standard deviation of the inference dataset before inputting these in the dataset script.

The dataset statistics used for the initial dataset were as follows: ```mean: 6.6374, std: 10.184```

The statistics for the transfer dataset were as follows: ```mean: 0.7294, std: 9.3929```

