# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:45:44 2024

@author: jenny
"""

# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load your data (ensure you update the path)
file_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/FeatureExtractedFiles/merged_data_train.csv'
data = pd.read_csv(file_path)

# Feature columns (based on your extracted features)
feature_columns = ['FirstPeakMagnitude', 'RMSValue', 'Mean', 'Max', 'Variance', 'Energy', 'FirstPeakFrequency', 'TimeOfFlight']

# Target label
X = data[feature_columns]  # Features
y = data['HumanLabel']  # Target (1 for human present, 0 for absent)

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (normalize the features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
scaler_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/scaler.pkl'
joblib.dump(scaler, scaler_path)

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model after hyperparameter tuning
best_rf_model = grid_search.best_estimator_

# Save the trained model to a file
joblib.dump(best_rf_model, 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/randomforest_model.pkl')


# Predictions on the test set
y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Human Absent', 'Human Present'],
            yticklabels=['Human Absent', 'Human Present'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Create a table for performance metrics
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Print the report table
print("\nPerformance Metrics:")
print(report_df)

# Visualize the performance metrics
metrics = {
    'Accuracy': accuracy,
    'Precision': report_df.loc['1', 'precision'],
    'Recall': report_df.loc['1', 'recall'],
    'F1 Score': report_df.loc['1', 'f1-score'],
}

metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

# Plotting the metrics
plt.figure(figsize=(10, 5))
#sns.heatmap(x='Metric', y='Value', data=metrics_df, palette='viridis')
sns.heatmap(metrics_df.set_index('Metric').T, annot=True, cmap='coolwarm', cbar=False, fmt='.2f')
plt.ylim(0, 1)  # Set the limit from 0 to 1 for metrics
plt.title('Random Forest Model Performance Metrics')
plt.grid(axis='y')
plt.show()








