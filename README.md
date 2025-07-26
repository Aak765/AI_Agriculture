# AI_Agriculture

# AI-Driven Precision Agriculture: Optimal Irrigation Prediction for Rice Farming

This project demonstrates an AI-driven approach to predict optimal irrigation schedules for farming, utilizing machine learning techniques. By analyzing various environmental and soil parameters, the model aims to provide data-backed insights for efficient water management in agriculture.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The core objective of this project is to develop a predictive model that suggests optimal irrigation amounts (in mm) for crops. This is achieved by training a Random Forest Regression model on a dataset containing critical agricultural parameters. The notebook AI_agriculture.ipynb covers data loading, preprocessing, model training, evaluation, and visualization.

*Note:* For demonstration purposes, the 'optimal_irrigation_mm' target variable in this notebook is synthetically generated based on other features. In a real-world application, this data would come from actual measured optimal irrigation levels.

## Features

* *Data Loading:* Handles loading agricultural data from a .csv or .xlsx file.
* *Data Preprocessing:*
    * Cleans column names.
    * Manages missing values (imputation with mean for numerical, mode for categorical).
    * Synthetically generates 'optimal_irrigation_mm' for demonstration.
    * Performs one-hot encoding for categorical features (soil_type, crop_type).
* *Machine Learning Model:* Implements a Random Forest Regressor for predicting irrigation.
* *Model Evaluation:* Assesses model performance using Mean Squared Error (MSE) and R-squared (R2).
* *Visualization:* Provides a scatter plot to visualize actual vs. predicted irrigation values.

## Dataset

The project expects a dataset (e.g., Rice_dataset.csv.xlsx) with the following columns (after cleaning):

* unnamed:_0 (potentially a date or index column)
* crop_type
* soil_type
* soil_ph
* temperature
* humidity
* wind_speed
* n (Nitrogen content)
* p (Phosphorus content)
* k (Potassium content)
* crop_yield
* soil_quality
* optimal_irrigation_mm (This is the target variable, synthetically generated in the notebook)

## Prerequisites

Before running the notebook, ensure you have the following Python libraries installed:

* pandas
* scikit-learn (sklearn)
* numpy
* matplotlib
* seaborn

You can install them using pip:

pip install pandas scikit-learn numpy matplotlib seaborn

Model Performance:
The Random Forest Regressor model is evaluated using:

Mean Squared Error (MSE): Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. Lower MSE indicates better fit.

R-squared (R2): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. R2 ranges from 0 to 1, with higher values indicating a better fit.

Example output from the provided notebook:

Model Performance:
  Mean Squared Error (MSE): 2.50
  R-squared (R2): 0.85
Visualization
A scatter plot is generated to visually compare the actual optimal irrigation values (simulated) against the values predicted by the model. A diagonal line represents perfect prediction, allowing for easy assessment of the model's accuracy.

Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the model, add new features, or enhance the documentation.

License
This project is open-sourced under the MIT License.
