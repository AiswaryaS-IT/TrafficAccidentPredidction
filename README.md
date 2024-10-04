# Traffic Accident Prediction with SVM

## Overview
This project predicts the likelihood of traffic accidents using a **Support Vector Machine (SVM)** model. The model takes real-time traffic data (such as traffic volume, road type, weather conditions, speed, and traffic density) as input and classifies the accident risk into "High" or "Low" based on several factors.

The project also provides an explanation for high-risk predictions by highlighting the factors contributing to the risk.

## Features
- **Traffic Accident Prediction**: Uses an SVM model trained on historical traffic data to predict accident likelihood.
- **Feature Explanation**: If the prediction is "High Risk," the system provides reasons such as **high speed**, **bad weather conditions**, or **high traffic density**.
- **Real-Time Data Input**: Accepts real-time traffic data for dynamic predictions.

## Tech Stack
- **Python** for the entire data pipeline and model.
- **pandas** for data manipulation.
- **scikit-learn** for model training, scaling, and prediction.

## Project Structure
- **trafficaccidentpredictionsvm.py**: The main Python file containing the code for the SVM model, data preprocessing, and prediction logic.

## Data
The project uses a dataset containing traffic accident records, including features such as:
- Traffic Volume
- Road Type (Highway, Urban, Rural)
- Weather Conditions (Clear, Rain, Fog)
- Speed of the Vehicle
- Traffic Density (Low, Medium, High)
- Accident Occurrence (Target Variable)
