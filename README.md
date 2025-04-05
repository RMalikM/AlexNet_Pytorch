# AlexNet PyTorch Implementation

## Overview
This repository contains an implementation of AlexNet using PyTorch for training, testing, and inference on the CIFAR-10 dataset.

## Repository Structure

    ALEXNET_PYTORCH/
    │── test_images/         # Directory for test images
    │── src/                # Source code directory
    │   │── __init__.py     # Init file
    │   │── dataset.py      # Data loading utilities
    │   │── inference.py    # Model inference script
    │   │── model.py        # AlexNet model implementation
    │   │── test.py         # Script for testing the trained model
    │   │── train.py        # Training script
    |── LICENSE             # License description
    |── README.md           # Readme file
    │── requirements.txt    # Required dependencies
    
## Requirements
Ensure you have the following dependencies installed before running the scripts:

    pip install -r requirements.txt

## Training the Model
To train AlexNet on CIFAR-10, run the following command:

    python src/train.py

### Training Parameters
The training script uses the following default parameters:

- **Number of Classes:** 10 (CIFAR-10 dataset)

- **Epochs:** 20

- **Batch Size:** 64

- **Learning Rate:** 0.001

The best model based on validation accuracy will be saved as best_model.pth.

### Training Workflow
1. Load the CIFAR-10 dataset and split it into training and validation sets.

2. Initialize the AlexNet model and configure it to run on a GPU (if available).

3. Define the loss function (`CrossEntropyLoss`) and the optimizer (`SGD`).

4. Train the model for a specified number of epochs, calculating loss and updating weights.

5. Evaluate the model on the validation set after each epoch.

6. Save the model if the validation accuracy improves.

### Testing the Model
Once trained, you can evaluate the model using:

    python src/test.py

### Running Inference
To perform inference on new images, use the `inference.py` script:

    python src/inference.py --image_path test_images/sample.jpg

### Model Implementation

The model.py script contains the AlexNet architecture defined using PyTorch. The model is adapted for CIFAR-10 by modifying the fully connected layers to match the dataset's 10 output classes.

### Dataset Handling

The dataset.py script includes utilities for loading and preprocessing the CIFAR-10 dataset, including splitting it into training and validation sets.

### Results

During training, validation accuracy is monitored, and the best model is saved. The final accuracy will be printed at the end of training.