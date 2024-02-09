# Import necessary libraries
"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    """
    Main function to load BrainVision data, apply a band-pass filter, plot the raw data,
    perform a Fast Fourier Transform (FFT) to analyze frequency components, 
    convert power spectral densities (PSDs) to dB, and plot the PSD.
    """

    # Change the current directory
    os.chdir(r"D:\CCEP_Data_Utrecht\sub-ccepAgeUMCU01\ses-1\ieeg")

    # Load the ECoG dataset & TSV files
    raw = mne.io.read_raw_brainvision((r"sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.vhdr"),preload=True)
    electrodes = pd.read_csv('sub-ccepAgeUMCU01_ses-1_electrodes.tsv', sep='\t')
    channels = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_channels.tsv', sep='\t')
    events = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_events.tsv', sep='\t')

    # Apply a band-pass filter on the raw data
    raw.filter(l_freq=0.1, h_freq=70)

    # # Plot the raw data
    # raw.plot()

    # Get the data from the Raw object
    data, times = raw[:]
    

    # # Perform a Fast Fourier Transform (FFT) to analyze frequency components
    # psds, freqs = mne.time_frequency.psd_array_welch(
    #     data, sfreq=raw.info['sfreq'], fmin=1., fmax=50.
    # )

    # Convert power spectral densities (PSDs) to dB
    # psds_db = 10 * np.log10(psds)

    # # Plot the PSD
    # plt.figure(figsize=(7, 4))
    # plt.plot(freqs, psds_db.mean(0).T, color='k', scalex=True, scaley=True)
    # plt.title('PSD (dB)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density (dB)')
    # plt.show()

if __name__ == "__main__":
    main()

# End of file
