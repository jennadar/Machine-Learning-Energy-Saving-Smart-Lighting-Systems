# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:50:26 2024

@author: jenny
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Load the CSV file (replace 'file_path' with the path to your uploaded dataset)
file_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Dataset/Human/adc_person_walk_human.csv'  # Provide the uploaded file path here
data = pd.read_csv(file_path)

# Extract the file name without extension to use it for the new file
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Let the user specify the location to store the new file
store_location = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Output/FeatureExtractedFiles'  # Provide the storage location path here

# Ensure the store location exists
if not os.path.exists(store_location):
    os.makedirs(store_location)

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

# Determine if the dataset is from a "human" or "nonhuman" folder
if 'human' in file_name:
    human_label = 1  # Label for human presence
elif 'NH' in file_name:
    human_label = 0  # Label for non-human presence
else:
    human_label = -1  # Undefined label if neither is found

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

# Calculate FFT for each row of ADC data and extract features
for index in range(adc_data.shape[0]):
    adc_values = adc_data.iloc[index, :]

    # Find the highest peak and its sample number
    peak_index = np.argmax(adc_values)  # Index of the highest peak
    highest_peak_value = adc_values[peak_index]  # Highest peak value
    sample_number_of_peak = peak_index + 1  # Sample number of the peak

    # Calculate Time of Flight (ToF = 2 * distance / speed of sound)
    # Ensure that `labelled_distance` is numeric, convert if necessary
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

# Save the extracted features into a DataFrame and add the human_label column
fft_features_df = pd.DataFrame(fft_features, columns=['FirstPeakMagnitude', 'RMSValue', 'Mean', 'Max', 'Variance', 'Energy', 'FirstPeakFrequency', 'TimeOfFlight'])

# Add human_label to the DataFrame
fft_features_df['HumanLabel'] = human_label

# Create a new file name for storing extracted features
new_file_name = f'{file_name}_extracted_features.csv'
full_store_path = os.path.join(store_location, new_file_name)
fft_features_df.to_csv(full_store_path, index=False)

print(f"Extracted features saved to: {full_store_path}")
