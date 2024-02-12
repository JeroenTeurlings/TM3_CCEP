"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
# Import necessary libraries
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nilearn import datasets, plotting
from nilearn.image import new_img_like

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
    ecog_channels = channels[channels['type'] == 'ECOG']['name'].tolist()
    events = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_events.tsv', sep='\t')
    events = events[['sample_start', 'trial_type']]
    # Filter the DataFrame
    events = events[events['trial_type'] == 'electrical_stimulation']

    # Change 'electrical_stimulation' to 1
    events['trial_type'] = events['trial_type'].replace('electrical_stimulation', 1)

    events.insert(loc=1, column='Zeros', value=np.zeros(events.shape[0], dtype=int))
    # Create an array of integers
    array_of_integers = events.values.astype(int)

    raw_ecog = raw.pick_channels(ecog_channels)
    
    # Extract Epochs
    epochs = mne.Epochs(raw_ecog, array_of_integers, event_id=1, tmin=-0.1, tmax=0.1)
    # epochs.plot(block=True)
  
    # Extract CCEP amplitudes
    amplitudes = epochs[257].get_data().max(axis=2) - epochs[0].get_data().min(axis=2)
    amplitudes_grid = amplitudes[:, :48].reshape(6, 8)
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(amplitudes_grid, cmap='viridis')
    # plt.show()

    # Apply a band-pass filter on the raw data
    raw.filter(l_freq=0.1, h_freq=70)

    # Plot the raw data
    # raw.plot()

    # Get the data from the Raw object
    data, times = raw[:]
    
    amplitudes = amplitudes[:, :48].reshape(-1, 1)
    destrieux_atlas = datasets.fetch_atlas_destrieux_2009()
    electrodes_destrieux = electrodes[['name', 'Destrieux_label']][:48]
    amplitude_map = np.zeros_like(destrieux_atlas['maps'], dtype=float)
    print(amplitude_map.shape)
    for electrode, amplitude in zip(electrodes_destrieux['Destrieux_label'], amplitudes):
        amplitude_map[destrieux_atlas['maps'] == electrode] = amplitude
        
    amplitude_img = new_img_like(destrieux_atlas['maps'], amplitude_map)
    print(amplitudes.shape) 
    print(electrodes_destrieux['Destrieux_label'].shape)
    print(amplitude_map.shape)
    print(destrieux_atlas['maps'])
    
    # Plot the amplitude data on the Destrieux atlas
    plotting.plot_stat_map(
        amplitude_img,
        bg_img=destrieux_atlas['maps'],
        cut_coords=(0, 0, 0),
        draw_cross=False,
        title="CCEP amplitude data on Destrieux atlas"
    )
    plotting.show()
    
    
    
if __name__ == "__main__":
    main()

# End of file
