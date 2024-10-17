# Import necessary libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert

def plot_signal(csv_path):
    # Load dataset from CSV into a DataFrame, selecting specific columns
    df = pd.read_csv(csv_path,header=None)
    df = df.iloc[:, 16:]

    # Iterate over each row in the DataFrame (each signal)
    for i in range(len(df.index)):
        f = df.iloc[i, :]  # Extract the signal from the current row

        # Signal processing parameters
        n = len(f)  # Length of the signal
        Fs = 1953125  # Sampling frequency in Hz
        dt = 1 / Fs
        time = np.arange(n)  # Time array for plotting

        # Fourier Transform and Power Spectral Density (PSD) calculation
        fhat = np.fft.fft(f, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        L = np.arange(1, n // 2, dtype='int')

        # Filter the signal by setting a threshold on the PSD
        indices = PSD > 1.5  # Threshold for filtering
        PSDclean = PSD * indices  # Filtered PSD
        fhat = indices * fhat  # Filtered Fourier Transform
        ffilt = np.fft.ifft(fhat)  # Inverse FFT to get the filtered signal

        # Hilbert Transform to compute the analytical signal and its envelope
        analytical_signal = hilbert(ffilt.real)
        env = np.abs(analytical_signal)  # Envelope of the analytical signal

        # Peak detection in the envelope
        x, _ = find_peaks(env, distance=n)

        # Calculate start time and end time based on peak positions
        for peak_position in x:
            start_time = max(0, peak_position - 2000)  # Ensure start time doesn't go below 0
            end_time = min(n, peak_position + 2000)     # Ensure end time doesn't exceed signal length

            # Plotting the filtered signal with envelope and peaks
            fig, axs = plt.subplots(2, 1, figsize=(10, 14))  # Create two subplots vertically

            # Plot 1: Original Filtered Signal with Envelope and Peaks
            axs[0].plot(time, ffilt, label='Filtered Signal')
            axs[0].plot(time, env, label='Envelope')
            axs[0].plot(x, env[x], "x", label='Peaks', color='red')
            axs[0].set_xlim(time[0], time[-1])
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')
            axs[0].legend()
            axs[0].set_title('Filtered Signal with Envelope and Peaks')

            # Plot 2: Zoomed-in Filtered Signal with Envelope and Peaks
            axs[1].plot(time, ffilt, label='Filtered Signal')
            axs[1].plot(time, env, label='Envelope')
            axs[1].plot(x, env[x], "x", label='Peaks', color='red')
            axs[1].set_xlim(start_time, end_time)  # Set x-axis limits based on start and end times
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Amplitude')
            axs[1].legend()
            axs[1].set_title('Zoomed-in Filtered Signal with Envelope and Peaks')

            plt.tight_layout()  # Adjust layout to prevent overlapping

            # Save the plot as an image file
            fig.savefig(f'static/plot.png')  # Save the plot with index i as 'plot_i.png' in the 'static' directory
            plt.close(fig)  # Close the figure to free up memory

        # Stop after the first signal due to the break statement
        break



csv_path = 'D:/jenny/Documents/FAUS_Study/Sem4/individual/1527226_Individual_Project/Dataset/Human/adc_person_walk_human.csv'
plot_signal(csv_path)