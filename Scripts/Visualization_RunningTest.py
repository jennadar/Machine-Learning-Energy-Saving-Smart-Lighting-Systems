# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:54:16 2024

@author: jenny
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve

# Load the test data
file_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/FeatureExtractedFiles/merged_data_test.csv'
test_data = pd.read_csv(file_path)

# Extract features (replace with the real feature column names)
feature_columns = ['FirstPeakMagnitude', 'RMSValue', 'Mean', 'Max', 'Variance', 'Energy', 'FirstPeakFrequency', 'TimeOfFlight']
X_test = test_data[feature_columns]
y_test = test_data['HumanLabel']  # Ensure y_test is defined here

# Load the saved Random Forest model
model_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/rf_model.pkl'
model = joblib.load(model_path)
#model = joblib.load('random_forest_model.pkl')

# Scale the features (ensure you apply the same scaler used during training)
# Save the scaler for later use
scaler_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/scaler_rf.pkl'
scaler = joblib.load(scaler_path)  # Assuming you saved the scaler too


X_test_scaled = scaler.transform(X_test)

# Make predictions using the loaded model
y_proba = model.predict(X_test_scaled)

# Set the optimal threshold (based on earlier calculations)
optimal_threshold = 0.89  # You can replace this with the actual optimal threshold calculated

# Predictions based on the optimal threshold
y_pred = (y_proba >= optimal_threshold).astype(int)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1 scores for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)

# Find the threshold that gives the maximum F1 score
optimal_idx = f1_scores.argmax()
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision', color='b')
plt.plot(thresholds, recall[:-1], label='Recall', color='g')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.4f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend(loc='best')
plt.show()

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot the actual values
axs[0].plot(y_test, color='blue', label='Actual Human Presence', linestyle='-', marker='o', markersize=2)
axs[0].set_title('Actual Human Presence', fontsize=16)
axs[0].set_ylabel('Light Status (1 = On, 0 = Off)', fontsize=14)
axs[0].legend()
axs[0].grid()

# Plot the predicted values
axs[1].plot(y_pred, color='red', label='Predicted Human Presence ', linestyle='--', marker='x', markersize=2)
axs[1].set_title('Predicted Human Presence ', fontsize=16)
axs[1].set_xlabel('Sample Index', fontsize=14)
axs[1].set_ylabel('Light Status (1 = On, 0 = Off)', fontsize=14)
axs[1].legend()
axs[1].grid()

# Add performance metrics text box
metrics_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
plt.figtext(0.15, 0.95, metrics_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Show the plot
plt.tight_layout()
plt.show()


