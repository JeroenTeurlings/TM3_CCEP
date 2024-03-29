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
from matplotlib import colormaps
import seaborn as sns
import pandas as pd
import scipy

# paths to mne datasets - FreeSurfer subject
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# Stimulus pair index to analyze
EPOCH_INDEX = 10

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
    # Make visualisation of available electrode locations
    plot_electrodes(electrodes, raw_ecog)
    # Make epochs
    epochs = make_epochs(raw_ecog, events)
    # #Plot the epochs
    # plot_epochs(epochs)
    amplitudes, latencies = find_ccep_peaks(epochs)
    # Plot the CCEP amplitude
    plot_ccep_amplitude(amplitudes, raw_ecog)
    # Plot the CCEP latency
    plot_ccep_latency(latencies, raw_ecog)
    # Plot gamma power
    plot_ccep_gamma(epochs, raw_ecog)
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
    os.chdir(r"D:\CCEP_Data_Utrecht\sub-ccepAgeUMCU02\ses-1\ieeg")
    # Subject
    subject = 'sub-ccepAgeUMCU02'
    # Session
    session = 'ses-1'
    # Task
    task = 'task-SPESclin'
    # Run
    run = 'run-021804'

    # Load the ECoG dataset & TSV files
    raw = mne.io.read_raw_brainvision((
        f'{subject}_{session}_{task}_{run}_ieeg.vhdr'), preload=True)
    electrodes = pd.read_csv(f'{subject}_{session}_electrodes.tsv', sep='\t')
    channels = pd.read_csv(f'{subject}_{session}_{task}_{run}_channels.tsv',
                           sep='\t')
    events = pd.read_csv(f'{subject}_{session}_{task}_{run}_events.tsv', sep='\t')
    print(raw.info)
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
    # raw_ecog.filter(l_freq=0.1, h_freq=40, method = 'iir')

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

    # fig = plot_alignment(
    #     raw_ecog.info,
    #     trans="fsaverage",
    #     subject="fsaverage",
    #     subjects_dir=subjects_dir,
    #     surfaces="pial",
    #     show_axes=True,
    #     ecog = True,
    #     sensor_colors=(1.0, 1.0, 1.0, 0.5),
    #     coord_frame="auto"
    # )
    # mne.viz.set_3d_view(fig, azimuth=180, elevation=90, focalpoint="auto", distance="auto")
    # xy, im = snapshot_brain_montage(fig, raw_ecog.info)

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
                            baseline=(None, -0.1), preload=True)
            stim_pair = list(event_id.keys())[i]
            stim_pair_split = stim_pair.split('-')
            epoch.drop_channels(stim_pair_split)
            print('\033[1m' + f'Analyzing stimpair {stim_pair}' + '\033[0m')
            epochs.append(epoch)
        except ValueError:
            print('\033[1m' + f'No epochs found for {stim_pair}' + '\033[0m')
    epochs[EPOCH_INDEX].plot_image(picks = [0], title='Averaged CCEP Response')
    evoked = epochs[EPOCH_INDEX].average()
    # evoked.plot(spatial_colors=True, gfp=True, time_unit='s')
    # evoked.plot_joint()
    return epochs

def plot_epochs(epochs):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epochs[0].plot(block=True, title="Epochs")

def find_ccep_peaks(epochs):
    """
    Function to find the peaks of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = epochs[EPOCH_INDEX].copy()
    amplitudes = []
    latencies = []

    for channels in range(len(selected_epoch.ch_names)):
        epoch_averaged = np.mean(selected_epoch.copy().get_data(picks=[channels], tmin=0.01, tmax=0.09), axis = 0).squeeze()
        epoch_sd = np.std(selected_epoch.get_data(picks=[channels], tmin=-2, tmax=-0.1))
        if epoch_sd < 50/1000000:
            epoch_sd = 50/1000000
        amplitudes_index, _ = scipy.signal.find_peaks(epoch_averaged,
                                                   distance= 200)
        plt.plot(epoch_averaged)
        plt.plot(amplitudes_index, epoch_averaged[amplitudes_index], "x")
        plt.plot(np.zeros_like(epoch_averaged), "--", color="gray")
        plt.show()
        # if len(amplitudes_index[0]) == 0:
        #     amplitudes_index = [[0]]
        # If the peak is at the beginning or end of the epoch, make amplitude 0
        print(amplitudes_index)
        if amplitudes_index == 0 or amplitudes_index == len(epoch_averaged)-1:
            amplitude = 0
        else:
            amplitude = epoch_averaged[amplitudes_index]
        # Print if amplitude is below threshold
        if amplitude < 2.6*epoch_sd:
            print(f'Channel {channels} has an amplitude of {amplitude} which is below the threshold of {epoch_sd*2.6}')
        else:
            print('Amplitude correct')
        latency = selected_epoch.times[amplitudes_index] + 2  # add 2 seconds to account for measurement window
        print(f'Channel {channels} has a peak at {latency} seconds with an amplitude of {amplitude}')
        amplitudes.append(amplitude)
        latencies.append(latency)

    return amplitudes, latencies

def plot_ccep_amplitude(amplitudes, raw_ecog):
    """
    Function to plot the CCEP amplitude.
    """
    # convert to dataframe
    amplitudes = pd.DataFrame(amplitudes)

    # scale amplitude values to be between 0 and 1, then map to colors
    amplitudes -= amplitudes.min()
    amplitudes /= amplitudes.max()
    rgba = colormaps.get_cmap("Reds")
    sensor_colors = np.array(amplitudes.map(rgba).values.tolist(), float)
    stim_color = np.array([[[1, 0.988, 0.216, 1]],
                           [[1, 0.988, 0.216, 1]]]) # yellow
    # Insert stimulation pair color at EPOCH_INDEX
    sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

    # plot the brain with the electrode colors for amplitude
    fig = mne.viz.plot_alignment(
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
    xy, im = mne.viz.snapshot_brain_montage(fig, raw_ecog.info)

def plot_ccep_latency(latencies, raw_ecog):
    """
    Function to plot the CCEP latency.
    """
    # convert to dataframe
    latencies = pd.DataFrame(latencies)

    # scale latency values to be between 0 and 1, then map to colors
    latencies -= latencies.min()
    latencies /= latencies.max()
    rgba = colormaps.get_cmap("Blues_r")
    sensor_colors = np.array(latencies.map(rgba).values.tolist(), float)
    stim_color = np.array([[[1, 0.988, 0.216, 1]],
                           [[1, 0.988, 0.216, 1]]]) # yellow
    # Insert stimulation pair color at EPOCH_INDEX
    sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

    # plot the brain with the electrode colors for latency
    fig = mne.viz.plot_alignment(
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
    xy, im = mne.viz.snapshot_brain_montage(fig, raw_ecog.info)

def plot_ccep_gamma(epochs, raw_ecog):
    """ 
    Function to plot the gamma power.
    """
    # Make a copy of the epoch to avoid modifying the original
    selected_epoch = epochs[EPOCH_INDEX].copy()
    # Extract gamma power    
    gamma_power_t = selected_epoch.copy().filter(30, 90).apply_hilbert(envelope=True)
    gamma_power_t = gamma_power_t.get_data().mean(axis=0)
    gamma_power_at_15s = pd.DataFrame(gamma_power_t).loc[:,0]

    # scale values to be between 0 and 1, then map to colors
    gamma_power_at_15s -= gamma_power_at_15s.min()
    gamma_power_at_15s /= gamma_power_at_15s.max()
    rgba = colormaps.get_cmap("viridis")
    sensor_colors = np.array(gamma_power_at_15s.map(rgba).values.tolist(), float)
    stim_color = np.array([[1, 0, 0, 1],
                           [1, 0, 0, 1]]) # red
    # Insert stimulation pair color at EPOCH_INDEX
    sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

    fig = mne.viz.plot_alignment(
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
    xy, im = mne.viz.snapshot_brain_montage(fig, raw_ecog.info)

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
