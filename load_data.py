"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
# Import necessary libraries
import os
import mne
from mne.epochs import Epochs
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
    # # Plot the raw data
    # plot_raw(raw)    
    # Preprocess the raw data into epochs
    epochs = preprocess(raw, channels, events)
    # #Plot the PSD
    # plot_psd(raw)
    # #Plot the epochs
    # plot_epochs(epochs)
    # #Map the amplitude of the CCEP response
    # amplitudes = map_amplitude(epochs)
    # #Map the CCEP response on the brain
    # map_overlay(amplitudes, electrodes, raw)

def load_data():
    # Change the current directory to the directory where the data is located
    os.chdir(r"D:\CCEP_Data_Utrecht\sub-ccepAgeUMCU01\ses-1\ieeg")

    # Load the ECoG dataset & TSV files
    raw = mne.io.read_raw_brainvision((
        'sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.vhdr'), preload=True)
    electrodes = pd.read_csv('sub-ccepAgeUMCU01_ses-1_electrodes.tsv', sep='\t')
    channels = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_channels.tsv',
                           sep='\t')
    events = pd.read_csv('sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_events.tsv', sep='\t')

    return raw, electrodes, channels, events

def plot_raw(raw):
    """
    Function to plot the raw data.
    """
    # Plot the raw data
    raw.plot(block=True, title="Raw Data")
    
def preprocess(raw, channels, events):
    # Filter out the ECoG channels in channels.tsv
    ecog_channels = channels[channels['type'] == 'ECOG']['name'].tolist()

    # Pick ECoG channels in raw data
    orig_raw = raw.copy()
    raw_ecog = raw.pick(ecog_channels)

    # Apply a band-pass filter to raw data
    raw_ecog.filter(l_freq=1, h_freq=40)

    # Extract CCEP events from events.tsv
    events_ccep = events[['sample_start', 'trial_type', 'electrical_stimulation_site']]
    events_ccep = events_ccep[events_ccep['trial_type'] == 'electrical_stimulation']
    events_ccep = events_ccep.drop(columns = ['trial_type'])
    
    # HOE MAAK JE VAN DEZE DATA EEN EVENT DICTIONARY? VERVANG STIMULATION SITE DOOR EVENT ID
    
    # Event dictionary with event_id being stimulated electrode?
    event_id = dict({'PT01-PT02': 0,  'PT03-PT02': 1,  'PT03-PT04': 2,  'PT05-PT04': 3,
                     'PT05-PT06': 4,  'PT07-PT06': 5,  'PT07-PT08': 6,  'PT09-PT08': 7,
                     'PT09-PT10': 8,  'PT11-PT10': 9,  'PT11-PT12': 10, 'PT13-PT12': 11,
                     'PT13-PT14': 12, 'PT15-PT14': 13, 'PT15-PT16': 14, 'PT17-PT16': 15,
                     'PT17-PT18': 16, 'PT19-PT18': 17, 'PT19-PT20': 18, 'PT21-PT20': 19,
                     'PT21-PT22': 20, 'PT23-PT22': 21, 'PT23-PT24': 22, 'PT25-PT24': 23,
                     'PT25-PT26': 24, 'PT27-PT26': 25, 'PT27-PT28': 26, 'PT29-PT28': 27,
                     'PT29-PT30': 28, 'PT31-PT30': 29, 'PT31-PT32': 30, 'PT33-PT32': 31,
                     'PT33-PT34': 32, 'PT35-PT34': 33, 'PT35-PT36': 34, 'PT37-PT36': 35,
                     'PT37-PT38': 36, 'PT39-PT38': 37, 'PT39-PT40': 38, 'PT41-PT40': 39,
                     'PT41-PT42': 40, 'PT43-PT42': 41, 'PT43-PT44': 42, 'PT45-PT44': 43,
                     'PT45-PT46': 44, 'PT47-PT46': 45, 'PT47-PT48': 46, 'F49-F50': 47,
                     'F51-F50': 48,   'F51-F52': 49,   'F53-F52': 50})

    # Change 'electrical_stimulation' to integer 1, add a column of zeros, make integer array
    events_ccep['electrical_stimulation_site'] = events_ccep['electrical_stimulation_site'].replace(event_id)
    events_ccep.insert(loc=1, column='Zeros', value=np.zeros(events_ccep.shape[0], dtype=int))
    events_ccep = events_ccep.values.astype(int)

    for i in range(0, len(event_id)):
        # Extract CCEP Epochs
        try:
            epochs = Epochs(raw_ecog, events_ccep, event_id=i, tmin=-2, tmax=2, baseline=(-2, -1), preload=True)
            stim_pair = list(event_id.keys())[i]
            stim_pair_split = stim_pair.split('-')
            epochs.drop_channels(stim_pair_split)
            print('\033[1m' + f'analyzing stimpair {stim_pair}' + '\033[0m')
            epochs.plot_image(title=f'Averaged CCEP Response for stimpair {stim_pair}')
        except ValueError:
            print('\033[1m' + f'No epochs found for {stim_pair}' + '\033[0m')
    return epochs

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
    plotting.plot_stat_map(amplitude_image, bg_img=destrieux_atlas.maps,
                           threshold=0.1, colorbar=True)
    plotting.show()

if __name__ == "__main__":
    main()

# End of file
