# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:07:10 2024

@author: jenny
"""

import os
import pandas as pd

# Define the folder containing the feature-extracted files
folder_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/FeatureExtractedFiles'

# Define the columns you want in the final merged file
columns = ['FirstPeakMagnitude', 'RMSValue', 'Mean', 'Max', 'Variance', 'Energy', 'FirstPeakFrequency', 'TimeOfFlight', 'HumanLabel']

# Create an empty DataFrame to hold all the merged data
merged_data = pd.DataFrame(columns=columns)

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Ensure we're only reading CSV files
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        file_data = pd.read_csv(file_path)
        
        # Ensure that the DataFrame has the expected columns before appending
        if set(columns).issubset(file_data.columns):
            # Append the data to the merged_data DataFrame
            merged_data = pd.concat([merged_data, file_data[columns]], ignore_index=True)
        else:
            print(f"Warning: {filename} does not have the required columns")

# Write the merged data to a new CSV file

output_file_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/FeatureExtractedFiles/merged_data_train.csv'
merged_data.to_csv(output_file_path, index=False)

print(f"Merged data saved to {output_file_path}")
