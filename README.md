# Deep-Learning-Model
Diabetes or Not - DLM

# PIMA Indians Diabetes Prediction Model

## Overview
This project involves creating a neural network model using Keras to predict diabetes among PIMA Indian individuals. The model is trained and evaluated on the PIMA Indians Diabetes dataset.

## Dependencies
- Python
- NumPy
- Keras
- scikit-learn
- Matplotlib

## Dataset
The dataset used is the 'pima-indians-diabetes.csv' file, which includes data for predicting diabetes based on certain health metrics.

## Model Architecture
The model is a Sequential neural network with the following layers:
- Dense layer with 12 nodes (ReLU activation)
- Dense layer with 6 nodes (ReLU activation)
- Dense layer with 1 node (Sigmoid activation)

## Training
The model is trained for 120 epochs with a batch size of 32. A portion of the training data is used for validation.

## Evaluation
The model's performance is evaluated on a separate test set, and accuracy metrics are provided. Loss and accuracy curves are plotted for both training and validation phases.

## Usage
Run the script to train the model and evaluate its performance on the test data. Plots for training and validation metrics will be displayed for analysis.

