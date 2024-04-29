# LSTM-Based Cryptocurrency Price Prediction

## Overview
This Python script implements an LSTM (Long Short-Term Memory) neural network model for predicting cryptocurrency price movements based on historical data and technical indicators. The model is trained on a dataset containing cryptocurrency price and indicator values, and it predicts the direction of the price movement for the next 5-minute interval.

## Features
- **Data Preprocessing**: The script preprocesses the input data, including scaling numerical features and reshaping for model compatibility.
- **LSTM Model Definition**: It defines an LSTM neural network model using the Keras API, comprising multiple LSTM layers followed by a dense output layer.
- **Model Training**: The model is trained on the preprocessed dataset using a specified number of epochs and batch size.
- **Model Evaluation**: After training, the script evaluates the model's performance on a validation set, calculating metrics such as accuracy, precision, recall, and F1-score.
- **Prediction**: Using the trained model, the script predicts the direction (positive or negative) of the next 5-minute price movement and provides a timestamp for the prediction.

## Usage
1. **Data Preparation**: Prepare a CSV file containing historical cryptocurrency price and indicator data.
2. **Data Loading and Preprocessing**: Modify the script to load your dataset and preprocess it according to your requirements.
3. **Model Configuration**: Adjust the LSTM model architecture, including the number of LSTM layers, units, activation functions, etc., as needed.
4. **Model Training**: Run the script to train the LSTM model on your preprocessed dataset.
5. **Model Evaluation**: Evaluate the trained model's performance using validation metrics printed in the console.
6. **Prediction**: Obtain predictions for future price movements based on the trained model.

## Dependencies
- Python 3.9.18
- pandas: Data manipulation library for handling datasets.
- numpy: Numerical computing library for array manipulation.
- scikit-learn: Machine learning library for data preprocessing and evaluation.
- tensorflow.keras: High-level neural networks API based on TensorFlow for building and training deep learning models.

## Note
- This script serves as a demonstration of LSTM-based cryptocurrency price prediction and should be customized and extended according to specific use cases and requirements.
- Ensure that you have sufficient historical data and relevant indicators for training a reliable prediction model.

## Contact
For any inquiries or support, please contact arman13m99@gmail.com

