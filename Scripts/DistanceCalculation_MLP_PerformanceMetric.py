# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:01:44 2024

@author: jenny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Load the CSV file
original_file_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Dataset/Human/adc_person_steady_human.csv'
data = pd.read_csv(original_file_path)

# Helper function to forcefully convert strings to floats
def force_convert_to_float(value):
    try:
        # Attempt to convert to float
        return float(value)
    except ValueError:
        # If conversion fails, strip non-numeric characters and try again
        value = ''.join(filter(lambda x: (x.isdigit() or x == '.' or x == '-'), str(value)))
        try:
            return float(value)
        except ValueError:
            # If it still fails, return a default value like 0 or np.nan
            print(f'Stiil unable to convert: {value}')
            return 0  # or np.nan if you want to ignore invalid data

# Use the 10th column as the label for distance
y = data.iloc[:, 10]

# Constants for distance calculation
speed_of_sound_meters = 343  # Speed of sound in air (m/s)

# Extract ADC data (from the 17th column onward)
adc_data = data.iloc[:, 16:]

# Number of samples for each reading
num_samples = adc_data.shape[1]

# Lists to store results for MLP training
fft_features = []

# Calculate FFT for each row of ADC data and extract features
for index in range(adc_data.shape[0]):
    adc_values = adc_data.iloc[index, :]

    # Find the highest peak and its sample number
    peak_index = np.argmax(adc_values)  # Index of the highest peak
    highest_peak_value = adc_values[peak_index]  # Highest peak value
    sample_number_of_peak = peak_index + 1  # Sample number of the peak

    # Calculate Time of Flight (ToF = 2 * distance / speed of sound)
    labelled_distance = force_convert_to_float(y.iloc[index])
    time_of_flight = (2 * labelled_distance) / speed_of_sound_meters

    # Calculate sampling frequency
    sampling_frequency = sample_number_of_peak / time_of_flight

    # Perform FFT using the calculated sampling frequency
    fft_values = fft(adc_values)
    fft_magnitude = np.abs(fft_values)[:num_samples // 2]  # Keep only positive frequencies

    # Frequency bins using the calculated sampling frequency
    freq_bins = fftfreq(num_samples, d=1/sampling_frequency)[:num_samples // 2]

    # Find the first significant peak frequency
    peak_indices, _ = find_peaks(fft_magnitude, height=0)  # Find peaks
    if len(peak_indices) > 0:
        first_peak_index = peak_indices[0]
        first_peak_frequency = freq_bins[first_peak_index]  # Frequency of the first peak
        first_peak_magnitude = fft_magnitude[first_peak_index]  # Magnitude of the first peak
    else:
        first_peak_frequency = 0  # No peak found, set to 0 or a default value
        first_peak_magnitude = 0  # no peak found
        
    
    
    # Extract features: mean, maximum, variance, energy, sampling frequency, and first peak frequency
    feature_mean = np.mean(fft_magnitude)
    feature_max = np.max(fft_magnitude)
    feature_variance = np.var(fft_magnitude)  # Variance of the FFT magnitudes
    feature_energy = np.sum(fft_magnitude ** 2)  # Energy of the signal
    rms_value = np.sqrt(np.mean(fft_magnitude**2))
    
    # Store features in a list
    fft_features.append([first_peak_magnitude, rms_value, feature_mean, feature_max, feature_variance, feature_energy, first_peak_frequency, time_of_flight])

# Convert the features and labels to numpy arrays
X = np.array(fft_features)  # Feature matrix
y = np.array(y)  # Labels

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [500, 1000],
}

grid_search = GridSearchCV(MLPRegressor(random_state=42, early_stopping=True), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate with cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print(f'Mean Cross-Validated MSE: {np.mean(cv_scores):.4f}')

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Binary classification: Convert continuous predictions into binary based on a threshold (e.g., median)
threshold = np.median(y)
y_test_binary = (y_test > threshold).astype(int)  # Binary classification (positive if above threshold)
y_pred_binary = (y_pred > threshold).astype(int)  # Predicted binary values

# Confusion matrix for binary classification
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)

# True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
tn, fp, fn, tp = conf_matrix.ravel()

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix (Binary Classification)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate classification metrics
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)

# Evaluate regression model metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a performance table
performance_metrics = {
    'Metric': ['Mean Squared Error (MSE)', 'R² Score', 'Precision', 'Recall', 'F1 Score', 'True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
    'Value': [mse, r2, precision, recall, f1, tp, tn, fp, fn]
}

performance_df = pd.DataFrame(performance_metrics)

# Display the performance table
print("\nPerformance Metrics:")
print(performance_df)

# Visualizing the performance table using a plot
plt.figure(figsize=(10, 4))
sns.heatmap(performance_df.set_index('Metric').T, annot=True, cmap='coolwarm', cbar=False, fmt='.2f')
plt.title('Performance Metrics Overview')
plt.show()

