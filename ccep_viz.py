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
from matplotlib import colormaps
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from PIL import Image

# Global settings for plotting
mne.viz.set_3d_backend('pyvista')

# Define the subject, session, task, and run
SUBJECT = 'sub-ccepAgeUMCU09'
SESSION = 'ses-1b'
TASK = 'task-SPESclin'
RUN = 'run-041737'

# paths to mne datasets - FreeSurfer subject
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# Stimulus pair index to analyze
EPOCH_INDEX = "P37-P38"

# Binary switches for plotting
PLOT_RAW = False
PLOT_ELECTRODES = False
PLOT_EPOCHS = False
PLOT_CCEP_AMPLITUDE = True
PLOT_MOV_AMPLITUDE = False
PLOT_CCEP_LATENCY = True
PLOT_CCEP_GAMMA = False

def main():
    """
    Main function to load BrainVision data, apply a band-pass filter, plot the raw data,
    perform a Fast Fourier Transform (FFT) to analyze frequency components, 
    convert power spectral densities (PSDs) to dB, and plot the PSD.
    """
    # Load the data
    raw, electrodes, channels, events = load_data()
    # Plot the raw data
    if PLOT_RAW:
        plot_raw(raw)
    # Preprocess the raw data into epochs
    raw_ecog, orig_raw = preprocess(raw, channels)
    # Set the montage
    set_montage(raw_ecog, electrodes)
    # Make visualisation of available electrode locations
    if PLOT_ELECTRODES:
        plot_electrodes(raw_ecog)
    # Make evoked
    evoked = make_epochs(raw_ecog, events)
    # Plot the epochs
    if PLOT_EPOCHS:
        plot_epochs(evoked)
    # Find the peaks of the CCEP response
    amplitudes, latencies = find_ccep_peaks(evoked)
    # Plot the CCEP amplitude
    if PLOT_CCEP_AMPLITUDE:
        plot_ccep_amplitude(amplitudes, raw_ecog)
    # Plot the CCEP amplitude over time
    if PLOT_MOV_AMPLITUDE:
        plot_mov_amplitude(evoked, raw_ecog)
    # Plot the CCEP latency
    if PLOT_CCEP_LATENCY:
        plot_ccep_latency(latencies, raw_ecog)
    # Plot gamma power
    if PLOT_CCEP_GAMMA:
        plot_ccep_gamma(evoked, raw_ecog)

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
    os.chdir(f"D:\\CCEP_Data_Utrecht\\{SUBJECT}\\{SESSION}\\ieeg")
    
    # Load the ECoG dataset & TSV files
    raw = mne.io.read_raw_brainvision((
        f'{SUBJECT}_{SESSION}_{TASK}_{RUN}_ieeg.vhdr'), preload=True)
    electrodes = pd.read_csv(f'{SUBJECT}_{SESSION}_electrodes.tsv', sep='\t')
    channels = pd.read_csv(f'{SUBJECT}_{SESSION}_{TASK}_{RUN}_channels.tsv',
                           sep='\t')
    events = pd.read_csv(f'{SUBJECT}_{SESSION}_{TASK}_{RUN}_events.tsv', sep='\t')
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

def set_montage(raw_ecog, electrodes):
    """
    Set the montage for the raw data.
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

def plot_electrodes(raw_ecog):
    """
    Function to plot the electrode locations.
    """
    fig = mne.viz.plot_alignment(
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
    xy, im = mne.viz.snapshot_brain_montage(fig, raw_ecog.info)

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

    epoch = Epochs(raw_ecog, events_ccep, event_id=event_id, tmin=-2, tmax=2,
                   baseline=(None, -0.1), preload=True)
    evoked = {site: epoch[site].average() for site in event_id.keys()}
    print(evoked[EPOCH_INDEX])
        
    # # initialize empty list to store epochs
    # evokeds = []

    # for i in range(0, len(event_id)):
    #     # Extract CCEP Epochs
    #     try:
    #         epoch = Epochs(raw_ecog, events_ccep, event_id=i, tmin=-2, tmax=2,
    #                         baseline=(None, -0.1), preload=True)
    #         stim_pair = list(event_id.keys())[i]
    #         stim_pair_split = stim_pair.split('-')
    #         epoch.drop_channels(stim_pair_split)
    #         print('\033[1m' + f'Analyzing stimpair {stim_pair}' + '\033[0m')
    #         evoked = epoch.average(picks=None, method='mean', by_event_type=True)
    #         evokeds.append(evoked)
    #     except ValueError:
    #         print('\033[1m' + f'No epochs found for {stim_pair}' + '\033[0m')
    # print(evokeds)
    # evokeds = evokeds[0]      
    # evokeds[EPOCH_INDEX].plot(spatial_colors=True, gfp=True, time_unit='s')
    # evokeds[EPOCH_INDEX].plot_joint()
    
    # evoked[EPOCH_INDEX].plot(spatial_colors=True, gfp=True, time_unit='s')
    # evoked[EPOCH_INDEX].plot_joint()
    
    return evoked

def plot_epochs(evoked):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    evoked[EPOCH_INDEX].plot(block=True, title="Epochs")

def find_ccep_peaks(evoked):
    """
    Function to find the peaks of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = evoked[EPOCH_INDEX].copy()
    amplitudes = []
    latencies = []

    for channels in range(len(selected_epoch.ch_names)):
        epoch_sd = np.std(selected_epoch.get_data(picks=[channels], tmin=-2, tmax=-0.1))
        epoch_data = selected_epoch.get_data(picks=[channels], tmin=0.009, tmax=0.100).squeeze()
        if epoch_sd < 50/1000000:
            epoch_sd = 50/1000000
        amplitudes_index, _ = scipy.signal.find_peaks(epoch_data,
                                                   distance= 200)
        plt.plot(epoch_data)
        plt.plot(amplitudes_index, epoch_data[amplitudes_index], "x")
        plt.plot(np.zeros_like(epoch_data), "--", color="gray")
        plt.show()
        # If the peak is at the beginning or end of the epoch, make amplitude 0
        if amplitudes_index == 0 or amplitudes_index == len(epoch_data)-1:
            amplitude = 0
        else:
            amplitude = epoch_data[amplitudes_index]
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
    
    stim_pair= EPOCH_INDEX.split('-')
    
    # Get index of stim_pair in raw_ecog
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names) if stim_pair[0] in s or stim_pair[1] in s]
    
    # scale amplitude values to be between 0 and 1, then map to colors
    non_stim_amplitudes = amplitudes.drop(stim_indices)
    non_stim_amplitudes -= non_stim_amplitudes.min()
    non_stim_amplitudes /= non_stim_amplitudes.max()
    rgba = colormaps.get_cmap("Reds")
    sensor_colors = np.array(non_stim_amplitudes.map(rgba).values.tolist(), float)
    stim_color = np.array([1, 1, 0, 1]) # yellow
    # # Insert stimulation pair color at EPOCH_INDEX
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

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

def plot_mov_amplitude(evoked, raw_ecog):
    """
    Function to plot the CCEP amplitude over time.
    """
    # Extract CCEP amplitudes
    selected_epoch = evoked[EPOCH_INDEX].copy()
    selected_epoch = selected_epoch.crop(tmin = -0.1, tmax = 0.2)
    all_amplitudes = []
    images = []

    for samples in range(len(selected_epoch.times)):
        print('Sample:', samples, 'of', len(selected_epoch.times))
        # Determine the amplitude of the CCEP response
        amplitudes = list()
        for channels in range(len(selected_epoch.ch_names)):
            # print('Channel:', channels, 'of', len(selected_epoch.ch_names))
            selected_epoch = selected_epoch.copy().get_data(picks=[channels])
            if samples < len(selected_epoch):
                amplitude = selected_epoch[samples]
                amplitudes.append(amplitude)
        all_amplitudes.extend(amplitudes)

    # Normalize all amplitudes
    all_amplitudes = pd.DataFrame(all_amplitudes)
    all_amplitudes -= all_amplitudes.min()
    all_amplitudes /= all_amplitudes.max()

    # Map all amplitudes to colors
    vmin = np.percentile(all_amplitudes, 1)
    vmax = np.percentile(all_amplitudes, 99)
    rgba = colormaps.get_cmap("RdBu_r")
    all_colors = np.array(all_amplitudes.map(lambda x: rgba((x - vmin) / (vmax - vmin))).values.tolist(), float)

    for samples in range(len(selected_epoch.times)):
        print('Plotting sample:', samples, 'of', len(selected_epoch.times))
        # Get the colors for the current sample
        sensor_colors = all_colors[samples * len(selected_epoch.ch_names):(samples + 1) * len(selected_epoch.ch_names)]

        # stim_color = np.array([[[1, 0.988, 0.216, 1]],
        #                     [[1, 0.988, 0.216, 1]]]) # yellow
        # # Insert stimulation pair color at EPOCH_INDEX
        # sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

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
            coord_frame="auto",
            verbose = False
        )
        # Set off_screen to True
        fig.plotter.off_screen = True
        mne.viz.set_3d_view(fig, azimuth=180, elevation=90, focalpoint="auto", distance="auto")
        xy, im = mne.viz.snapshot_brain_montage(fig, raw_ecog.info)
        pil_im = Image.fromarray(im)
        images.append(pil_im)
        # remove variable amplitude
        mne.viz.close_3d_figure(fig)

    image_one = images[0]
    os.chdir(r"C:\Users\jjbte\OneDrive\Documenten\TM3\Afstuderen\CCEP_GIF")
    image_one.save(f"CCEP_{EPOCH_INDEX}.gif", format="GIF", append_images=images[1:],
                   save_all=True, duration=100, loop=0)
    del images, image_one, pil_im, im, xy, fig, all_colors, all_amplitudes

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
    # stim_color = np.array([[[1, 0.988, 0.216, 1]],
    #                        [[1, 0.988, 0.216, 1]]]) # yellow
    # # Insert stimulation pair color at EPOCH_INDEX
    # sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

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

def plot_ccep_gamma(evoked, raw_ecog):
    """ 
    Function to plot the gamma power.
    """
    # Make a copy of the epoch to avoid modifying the original
    selected_epoch = evoked[EPOCH_INDEX].copy()
    # Extract gamma power    
    gamma_power_t = selected_epoch.copy().filter(30, 90).apply_hilbert(envelope=True)
    gamma_power_t = gamma_power_t.get_data().mean(axis=0)
    gamma_power_at_15s = pd.DataFrame(gamma_power_t).loc[:,0]

    # scale values to be between 0 and 1, then map to colors
    gamma_power_at_15s -= gamma_power_at_15s.min()
    gamma_power_at_15s /= gamma_power_at_15s.max()
    rgba = colormaps.get_cmap("viridis")
    sensor_colors = np.array(gamma_power_at_15s.map(rgba).values.tolist(), float)
    # stim_color = np.array([[1, 0, 0, 1],
    #                        [1, 0, 0, 1]]) # red
    # # Insert stimulation pair color at EPOCH_INDEX
    # sensor_colors = np.insert(sensor_colors, EPOCH_INDEX, stim_color, axis = 0)

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

if __name__ == "__main__":
    main()

# End of file
