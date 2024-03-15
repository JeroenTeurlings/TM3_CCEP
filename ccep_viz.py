"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
# Import necessary libraries
import os
import mne
from mne.epochs import Epochs
from mne.viz import plot_alignment, snapshot_brain_montage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import pandas as pd

# paths to mne datasets - FreeSurfer subject
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

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
    raw_ecog, orig_raw = preprocess(raw, channels)
    # Make scatterplot of available electrode locations
    plot_electrodes(electrodes, raw_ecog)
    # Make epochs
    epochs = make_epochs(raw_ecog, events)
    # #Plot the epochs
    # plot_epochs(epochs)
    amplitudes, times, sensor_colors = find_ccep_peaks(epochs, raw_ecog)
    # # Map the amplitude of the CCEP response
    # amplitudes = map_amplitude(epochs)

def load_data():
    """
    Loads the ECoG dataset and associated TSV files.

    This function changes the current directory to the directory where the data is located,
    and then loads the ECoG dataset and TSV files using the MNE library and pandas.

    Returns:
        raw (mne.io.Raw): The loaded ECoG dataset.
        electrodes (pandas.DataFrame): The loaded electrodes information.
        channels (pandas.DataFrame): The loaded channel information.
        events (pandas.DataFrame): The loaded event information.
    """
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

def preprocess(raw, channels):
    """
    Preprocesses the raw data by filtering ECoG channels, applying a band-pass filter,
    and extracting CCEP events.

    Args:
        raw (mne.io.Raw): The raw data.
        channels (pandas.DataFrame): DataFrame containing information about the channels.
        events (pandas.DataFrame): DataFrame containing information about the events.

    Returns:
        mne.Epochs: The preprocessed epochs.

    Raises:
        ValueError: If no epochs are found for a specific stimulation pair.

    """
    # Filter out the ECoG channels in channels.tsv
    ecog_channels = channels[channels['type'] == 'ECOG']['name'].tolist()

    # Pick ECoG channels in raw data
    orig_raw = raw.copy()
    raw_ecog = raw.pick(ecog_channels)

    # # Apply a band-pass filter to raw data
    # raw_ecog.filter(l_freq=1, h_freq=40, method = 'iir')

    return raw_ecog, orig_raw

def plot_electrodes(electrodes, raw_ecog):
    """
    Function to plot the electrode locations.
    """
    # Make electrode postions into a dictionary
    el_position = {row['name']: [row['x'], row['y'], row['z']] for index, row in electrodes.iterrows()}

    # Change position of electrodes to meters
    for key in el_position:
        el_position[key] = [i / 1000 for i in el_position[key]]

    # the coordinate frame of the montage
    montage = mne.channels.make_dig_montage(el_position, coord_frame='mni_tal')
    montage.add_mni_fiducials(subjects_dir)
    raw_ecog.set_montage(montage)

    fig = plot_alignment(
        raw_ecog.info,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=subjects_dir,
        surfaces="pial",
        show_axes=True,
        ecog = True,
        sensor_colors=(1.0, 1.0, 1.0, 0.5),
        coord_frame="auto"
    )
    mne.viz.set_3d_view(fig, azimuth=180, elevation=90, focalpoint="auto", distance="auto")
    xy, im = snapshot_brain_montage(fig, raw_ecog.info)
    
def make_epochs(raw_ecog, events):
    """
    Extracts CCEP events from the given events dataframe and creates epochs for each event.

    Parameters:
    raw_ecog (object): The raw ecog data.
    events (DataFrame): The events dataframe containing information about the events.

    Returns:
    list: A list of epochs, each representing a CCEP event.

    Raises:
    ValueError: If no epochs are found for a specific stimulation pair.

    """
    # Extract CCEP events from events.tsv
    events_ccep = events[['sample_start', 'trial_type', 'electrical_stimulation_site']]
    events_ccep = events_ccep[events_ccep['trial_type'] == 'electrical_stimulation']
    events_ccep = events_ccep.drop(columns = ['trial_type'])

    # Create a dictionary of event IDs extracted from events_ccep
    unique_stim_sites = events_ccep['electrical_stimulation_site'].unique()
    event_id = {site: i for i, site in enumerate(unique_stim_sites)}

    # Change 'electrical_stimulation' to integer 1, add a column of zeros, make integer array
    events_ccep['electrical_stimulation_site'] = events_ccep['electrical_stimulation_site'].replace(event_id)
    events_ccep.insert(loc=1, column='Zeros', value=np.zeros(events_ccep.shape[0], dtype=int))
    events_ccep = events_ccep.values.astype(int)

    # initialize empty list to store epochs
    epochs = []

    for i in range(0, len(event_id)):
        # Extract CCEP Epochs
        try:
            epoch = Epochs(raw_ecog, events_ccep, event_id=i, tmin=-2, tmax=2,
                            baseline=(-1, -0.1), preload=True)
            stim_pair = list(event_id.keys())[i]
            stim_pair_split = stim_pair.split('-')
            epoch.drop_channels(stim_pair_split)
            print('\033[1m' + f'Analyzing stimpair {stim_pair}' + '\033[0m')
            epochs.append(epoch)
        except ValueError:
            print('\033[1m' + f'No epochs found for {stim_pair}' + '\033[0m')
    epochs[0].plot_image(picks = [94], title='Averaged CCEP Response')
    
    return epochs

def plot_epochs(epochs):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epochs[0].plot(block=True, title="Epochs")

def find_ccep_peaks(epochs, raw_ecog):
    """
    Function to find the peaks of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = epochs[20].copy()
    amplitudes = []
    latencies = []
    
    for channels in range(len(selected_epoch.ch_names)):
        epoch_ccep = selected_epoch.get_data(picks=[channels], tmin=0.009, tmax=0.1)
        amplitude = epoch_ccep.max(axis=2)
        time = epoch_ccep.argmax(axis=2)
        latency = selected_epoch.times[time] + 2  # add 2 seconds to account for measurement window
        amplitudes.append(amplitude.mean())
        latencies.append(latency.mean())
    
    # convert to dataframe
    amplitudes = pd.DataFrame(amplitudes)
    latencies = pd.DataFrame(latencies)
    
    # scale values to be between 0 and 1, then map to colors
    amplitudes -= amplitudes.min()
    amplitudes /= amplitudes.max()
    # Add two values of 1 at the start of the dataframe to compensate for the first two channels
    amplitudes = pd.concat([pd.DataFrame([1, 1]), amplitudes], ignore_index=True)
    rgba = colormaps.get_cmap("Reds")
    sensor_colors = np.array(amplitudes.map(rgba).values.tolist(), float)
    print(sensor_colors)
    
    fig = plot_alignment(
        raw_ecog.info,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=subjects_dir,
        surfaces="pial",
        show_axes=True,
        ecog = True,
        sensor_colors=sensor_colors,
        coord_frame="auto"
    )
    mne.viz.set_3d_view(fig, azimuth=180, elevation=90, focalpoint="auto", distance="auto")
    xy, im = snapshot_brain_montage(fig, raw_ecog.info)
    
    
    return amplitudes, latencies, sensor_colors
    
def map_amplitude(epochs):
    """
    Function to map the amplitude of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = epochs[1].copy()
    amplitudes = selected_epoch.get_data(tmin = 0.009, tmax = 0.1).max(axis=2) - selected_epoch.get_data(tmin = 0.009, tmax = 0.1).min(axis=2)
    plt.figure()
    sns.heatmap(amplitudes, cmap='viridis')
    plt.show()

    return amplitudes

if __name__ == "__main__":
    main()

# End of file
