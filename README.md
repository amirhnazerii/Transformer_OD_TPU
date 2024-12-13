# Transformer_OD_TPU

Exploration of TPU Architectures for Optimized Transformer Performance in Image Detection of Drainage Crossings

![image](https://github.com/user-attachments/assets/26502408-11b3-4a8f-9578-5e31a16c8ac6)
(Carion et al. 2020)
![image](https://github.com/user-attachments/assets/13406859-337b-4907-aece-f68805006bad)



## Repository Structure

A working structure for a Dockerized application of this repository, with results, visualizations, and the dataset stored outside the code repository, should follow this structure. The result_folder and visualizations directories are created by running ```main.py``` and the notebooks under the visualization subdirectory.
```
/workspace/
├── Transformer_OD_TPU
│   ├── ScaleSim
│   │   └── Experiment Results
│   ├── datasets
│   ├── models
│   ├── notebooks
│   │   ├── data_preparation
│   │   ├── tests
│   │   └── visualization
│   └── util
|
├── processed_data
│   ├── initial_data
│   │   ├── annotations
│   │   ├── test
│   │   ├── train
│   │   └── validate
│   └── transfer_data
│       ├── annotations
│       └── test
|
├── result_folder
└── visualizations
```

## Data Preprocessing, Testing, and Visualization using Notebooks

Raw data were preprocessed and data statistics were extracted using the notebooks under the data_preparation subdirectory. Data are preprocessed and object jsons are created for each crop level to conform to the coco dataset spec. These notebooks should not be used on the preprocessed dataset, and are included to make reproduction from the initial raw dataset possible.

Testing performance is measured using the notebook under the tests subdirectory, and visualizations and graphs are generated using the visualization subdirectory.

## Running the Model in Docker

Ensure that the data folder and Transformer_OD_TPU repository are in the same work directory, ```/path/to/work/directory/```.

Navigate to the ```Transformer_OD_TPU``` repository and build the docker image by running:

```docker build -t transformer_od_tpu .```

Then, after building the Docker image, run a jupyter lab server using:

```docker run --gpus all -it -v /path/to/work/directory:/workspace/ -p 8888:8888 transformer_od_tpu```

Then, the program can be run using the following command in the ```Transformer_OD_TPU``` directory, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output --num_workers 0 --batch_size 8 --crop 400 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth```

This will run the script with default parameters, loading in pretrained object detection weights.

For the experiments performed in the accompanying paper 'Exploration of TPU Architectures for the Optimized Transformer in Drainage Crossing Detection', experiments were run using the above command, changing the ```--crop``` argument to 256, 400, 600, and 800.

To train from scratch, a binary DETR model can be built by passing ```num_classes 2``` as an argument, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output_scratch --num_workers 0 --batch_size 8 --crop 400 --num_classes 2```

## Dataset

In order to test model transferrability to an unseen watershed, we first partition the data into the initial dataset and the transfer dataset.

The initial dataset, which includes watersheds from NE, CA, and IL, is randomly paritioned into training (70%), validation (20%), and testing (10%) sets.

The transfer dataset, which includes data from the ND watershed, is used in its entirety to test model transferrability to unseen HRDEM data.

![image](https://github.com/user-attachments/assets/c1673d8c-c612-4905-9352-3491cc4d8f6c)


The initial and transfer dataset are normalized according to their corresponding dataset mean and standard deviation to simulate the model's use in inference. It is expected that for a given inference dataset, the user will first find the mean and standard deviation of the inference dataset before inputting these in the dataset script.

The dataset statistics used for the initial dataset were as follows: ```mean: 6.6374, std: 10.184```
The statistics for the transfer dataset were as follows: ```mean: 0.7294, std: 9.3929```

## Results
- Predictions:
![image](https://github.com/user-attachments/assets/d726725c-ec22-4340-80f6-aa3752bdd8ea)


![image](https://github.com/user-attachments/assets/0e07e026-4724-40f1-9145-20564fa01c59)

Model outputs from initial test set (top) and transfer learning test set (bottom). From left to right: original image size of 800×800 is cropped to 600×600, 400×400, and 256×256. Ground truth bounding boxes are shown in green and model predictions with confidence over 0.7 are shown in red.![image](https://github.com/user-attachments/assets/419ff5a8-b054-4ba3-bf57-fa53667b5d84)

###  Model Performance:

Precision and Recall on Initial and Transfer Datasets. Average Precision = AP, Average Recall = AR
![image](https://github.com/user-attachments/assets/410bda77-a196-4b10-bf55-8685e9f9bbe1)

![image](https://github.com/user-attachments/assets/8dcc983d-caf8-4447-bd6e-612de26a9278)




