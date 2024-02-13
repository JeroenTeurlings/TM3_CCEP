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
import nibabel as nib
from nilearn import datasets, plotting, image
from nilearn.image import new_img_like

def main():
    """
    Main function to load BrainVision data, apply a band-pass filter, plot the raw data,
    perform a Fast Fourier Transform (FFT) to analyze frequency components, 
    convert power spectral densities (PSDs) to dB, and plot the PSD.
    """
    # Load the data
    raw, electrodes, channels, events = load_data()
    # Preprocess the raw data into epochs
    epochs = preprocess(raw, channels, events)
    # Plot the raw data
    plot_raw(raw)
    # Plot the PSD
    plot_psd(raw)
    # Plot the epochs
    plot_epochs(epochs)
    # Map the amplitude of the CCEP response
    amplitudes = map_amplitude(epochs)
    # Map the CCEP response on the brain
    map_overlay(amplitudes, electrodes, raw)

def load_data():
    # Change the current directory to the directory where the data is located
    os.chdir(r"D:\CCEP_Data_Utrecht\sub-ccepAgeUMCU01\ses-1\ieeg")

    # Load the ECoG dataset & TSV files
    raw = mne.io.read_raw_brainvision(('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.vhdr'),
                                    preload=True)
    electrodes = pd.read_csv('sub-ccepAgeUMCU01_ses-1_electrodes.tsv', sep='\t')
    channels = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_channels.tsv', sep='\t')
    events = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_events.tsv', sep='\t')
    
    return raw, electrodes, channels, events

def preprocess(raw, channels, events):
    # Apply a band-pass filter to raw data
    raw.filter(l_freq=1, h_freq=70)

    # Filter out the ECoG channels in channels.tsv
    ecog_channels = channels[channels['type'] == 'ECOG']['name'].tolist()  

    # Pick ECoG channels in raw data
    raw_ecog = raw.pick(ecog_channels)

    # Extract CCEP events from events.tsv
    events_ccep = events[['sample_start', 'trial_type']]
    events_ccep = events_ccep[events_ccep['trial_type'] == 'electrical_stimulation']

    # Change 'electrical_stimulation' to integer 1, add a column of zeros, make integer array
    events_ccep['trial_type'] = events_ccep['trial_type'].replace('electrical_stimulation', 1)
    events_ccep.insert(loc=1, column='Zeros', value=np.zeros(events_ccep.shape[0], dtype=int))
    events_ccep = events_ccep.values.astype(int)

    # Extract CCEP Epochs
    epochs = mne.Epochs(raw_ecog, events_ccep, event_id=1, tmin=-0.1, tmax=0.1)
    
    return epochs

def plot_raw(raw):
    """
    Function to plot the raw data.
    """
    # Plot the raw data
    raw.plot(block=True, title="Raw Data")
    
def plot_epochs(epochs):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epochs.plot(block=True, title="Epochs")

def plot_psd(raw):
    """
    Function to perform a Fast Fourier Transform (FFT) to analyze frequency components, 
    convert power spectral densities (PSDs) to dB, and plot the PSD.
    """
    data = raw.get_data()
    
    # Perform a Fast Fourier Transform (FFT) to analyze frequency components
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info['sfreq'], fmin=1, fmax=50
    )

    # Convert power spectral densities (PSDs) to dB
    psds_db = 10 * np.log10(psds)

    # Plot the PSD
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psds_db.mean(0).T, color='k', scalex=True, scaley=True)
    plt.title('PSD (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.show()

def map_amplitude(epochs):
    """
    Function to map the amplitude of the CCEP response.
    """
    # Extract CCEP amplitudes
    amplitudes = epochs.get_data().max(axis=2) - epochs.get_data().min(axis=2)
    amplitudes_grid = amplitudes[0, :48].reshape(6, 8)
    plt.figure(figsize=(10, 6))
    sns.heatmap(amplitudes_grid, cmap='viridis')
    plt.show()
    
    return amplitudes

def map_overlay(amplitudes, electrodes, raw):
    """
    Function to overlay the CCEP response on the brain.
    """
    # # Overlay the CCEP response on the brain
    # # Create a Nifti image
    # img = new_img_like(raw, np.ones(raw.get_data().shape))
    # # Plot the CCEP response on the brain
    # display = plotting.plot_img(img, bg_img=False, cut_coords=(0, 0, 0), title="CCEP Response")
    # display.add_overlay(img, cmap='hot', threshold=0.5)   

    amplitudes = amplitudes[0, :48].reshape(-1, 1)
    destrieux_atlas = datasets.fetch_atlas_destrieux_2009()
    
    amplitudes = np.array([0.5, 0.7, 0.9])
    electrode_coords = np.array([[30, 40, 50], [60, 70, 20], [60, 75, 75]])
    
    amplitude_image = np.zeros([76, 93, 76] , dtype=float)
    for coord, amp in zip(electrode_coords, amplitudes):
        amplitude_image[tuple(coord)] = amp
    
    
    
    # electrodes_destrieux = electrodes[['name', 'Destrieux_label']][:48]
    # amplitude_map = np.zeros_like(destrieux_atlas['maps'], dtype=float)
    # print(amplitude_map.shape)
    # for electrode, amplitude in zip(electrodes_destrieux['Destrieux_label'], amplitudes):
    #     amplitude_map[destrieux_atlas['maps'] == electrode] = amplitude



    amplitude_image = image.new_img_like(destrieux_atlas.maps, amplitude_image)

    # Plot the atlas with the amplitude overlay
    plotting.plot_stat_map(amplitude_image, bg_img=destrieux_atlas.maps, threshold=0.1, colorbar=True)
    plotting.show()

if __name__ == "__main__":
    main()

# End of file
