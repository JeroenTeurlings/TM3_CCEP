"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
# Import necessary libraries
import os
import io
import time
import sys
import mne
from mne.epochs import Epochs
import numpy as np
from matplotlib import colormaps
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import scipy
from scipy.stats.mstats import winsorize
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
# src = mne.read_source_spaces(
#     subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
# )

# Stimulus pair index to analyze
STIM_PAIR = "P47-P48"

# Binary switches for plotting
PLOT_RAW = False
PLOT_ELECTRODES = True
PLOT_ELECTRODES_GIF = False
PLOT_EPOCHS = False
PLOT_PEAKS = False
PLOT_CCEP_AMPLITUDE = False
PLOT_MOV_AMPLITUDE = False # Computational heavy
THRESH = False
PLOT_CCEP_LATENCY = False
PLOT_CCEP_GAMMA = False
PLOT_MOV_GAMMA = False # Computational heavy

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
    evoked, epoch = make_epochs(raw_ecog, events)
    # Plot the epochs
    if PLOT_EPOCHS:
        plot_epochs(epoch)
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
    # Plot gamma power over time    
    if PLOT_MOV_GAMMA:
        plot_mov_gamma(evoked, raw_ecog)

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
    el_position = {row['name']: [row['x'], row['y'], row['z']] for index,
                   row in electrodes.iterrows()}

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
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage",
                  hemi="both",
                  surf="pial",
                  cortex="grey",
                  subjects_dir=subjects_dir,
                  background="white",
                  interaction="terrain",
                  show=True)
    brain.add_annotation("aparc.a2009s",
                         borders=False,
                         alpha=0.0)
    brain.add_sensors(raw_ecog.info,
                      trans="fsaverage",
                      ecog = True,
                      sensor_colors=(1.0, 1.0, 1.0, 0.5))
    
    if PLOT_ELECTRODES_GIF:
        screenshots = []
        for azimuth in range(0, 360, 1):
            brain.show_view(azimuth=azimuth,
                            elevation=90,
                            focalpoint="auto",
                            distance="auto")
            screenshots.append(Image.fromarray(brain.screenshot(mode="rgb", time_viewer=True)))
            
        os.chdir(r"C:\Users\jjbte\OneDrive\Documenten\TM3\Afstuderen\CCEP_GIF")
        screenshots[0].save("Electrodes_annot.gif", format="GIF",
                    append_images=screenshots[1:],
                    save_all=True, duration=50, loop=0)
    
    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")
    
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
    events_ccep['electrical_stimulation_site'] = \
        events_ccep['electrical_stimulation_site'].replace(event_id)
    events_ccep.insert(loc=1, column='Zeros', value=np.zeros(events_ccep.shape[0], dtype=int))
    events_ccep = events_ccep.values.astype(int)

    epoch = Epochs(raw_ecog, events_ccep, event_id=event_id, tmin=-2, tmax=2,
                   baseline=None, preload=True, verbose=False)
    evoked = {site: epoch[site].average() for site in event_id.keys()}

    # apply baseline correction for every evoked in evoked
    for site in evoked:
        evoked[site].apply_baseline(baseline = (None, -0.1), verbose = False)
    return evoked, epoch

def plot_epochs(epoch):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epoch[STIM_PAIR].plot(block=True, title="Epochs")

def find_ccep_peaks(evoked):
    """
    Function to find the peaks of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = evoked[STIM_PAIR].copy()
    amplitudes = []
    latencies = []

    for channels in range(len(selected_epoch.ch_names)):
        epoch_sd = np.std(selected_epoch.get_data(picks=[channels], tmin=-2, tmax=-0.1))
        epoch_data = selected_epoch.get_data(picks=[channels],
                                             tmin=0.009, 
                                             tmax=0.100).squeeze()*-1 # Invert data for N1 peak
        if epoch_sd < 50/1000000:
            epoch_sd = 50/1000000
        amplitudes_index, _ = scipy.signal.find_peaks(epoch_data,
                                                   distance= 200)
        amplitude = epoch_data[amplitudes_index]
        if PLOT_PEAKS:
            plt.plot(epoch_data)
            plt.plot(amplitudes_index, epoch_data[amplitudes_index], "x")
            plt.plot(np.zeros_like(epoch_data), "--", color="gray")
            plt.show()

        # Print if amplitude is below threshold
        if amplitude.size > 0 and amplitude < 2.6 * epoch_sd:
            if PLOT_PEAKS:
                print(f'Channel {channels} has an amplitude of {amplitude}' 
                      f' which is below the threshold of {epoch_sd*2.6}')
        else:
            if PLOT_PEAKS:
                print('Amplitude correct')
        # add 2 seconds to account for measurement window
        latency = selected_epoch.times[amplitudes_index] + 2
        if PLOT_PEAKS:
            print(f'Channel {channels} has a peak at {latency}'
                  f' seconds with an amplitude of {amplitude}')
        amplitudes.append(amplitude)
        latencies.append(latency)

    return amplitudes, latencies

def plot_ccep_amplitude(amplitudes, raw_ecog):
    """
    Function to plot the CCEP amplitude.
    """
    ## Define the colormap
    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    non_stim_amplitudes = np.delete(amplitudes, stim_indices)
    # Winsorize the data
    non_stim_amplitudes = winsorize(non_stim_amplitudes, limits=[0.01, 0.01])
    # convert to dataframe
    non_stim_amplitudes = pd.DataFrame(non_stim_amplitudes)
    # Take absolute value of amplitudes
    non_stim_amplitudes_norm = 2 * (non_stim_amplitudes - non_stim_amplitudes.min()) \
                                    / (non_stim_amplitudes.max() - non_stim_amplitudes.min()) - 1
    rgba = colormaps.get_cmap("Reds")
    sensor_colors = np.array(non_stim_amplitudes_norm.map(rgba).values.tolist(), float)
    stim_color = np.array([1, 1, 0, 1]) # yellow
    # Insert stimulation pair color at EPOCH_INDEX
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

    ## Plot the brain with the electrode colors for amplitude
    # plot the brain with the electrode colors for amplitude
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage",
                  hemi="both",
                  surf="pial",
                  cortex="grey",
                  subjects_dir=subjects_dir,
                  background="white",
                  interaction="terrain",
                  show=True)
    brain.add_annotation("aparc.a2009s",
                         borders=False,
                         alpha=0.8)
    brain.add_sensors(raw_ecog.info,
                      trans="fsaverage",
                      ecog = True,
                      sensor_colors=sensor_colors)
    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

def plot_mov_amplitude(evoked, raw_ecog):
    """
    Function to plot the CCEP amplitude over time.
    """
    # Initialize variables
    tmin = -0.1
    tmax = 0.2
    selected_epoch = evoked[STIM_PAIR].copy()
    selected_epoch = selected_epoch.crop(tmin = tmin, tmax = tmax)
    all_amplitudes = []
    images = []
    thresh = 2.6*50/1000000

    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    
    # Initialize brain
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage",
                  hemi="both",
                  surf="pial",
                  cortex="grey",
                  subjects_dir=subjects_dir,
                  background="white",
                  interaction="terrain",
                  show=True)
    brain.show_view(azimuth=180,
                    elevation=90,
                    focalpoint="auto",
                    distance="auto")
        
    brain.add_annotation("aparc.a2009s",
                            borders=False,
                            alpha=0.8,
                            remove_existing=True)

    ## Main loop
    for samples in range(len(selected_epoch.times)):
        # Progress bar
        j = (samples + 1) / len(selected_epoch.times)
        sys.stdout.write('\r')
        sys.stdout.write("Calculating sample amplitudes: [%-20s] %d%% " % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        # Determine the amplitude of the CCEP response
        amplitudes = []
        for channels in range(len(selected_epoch.ch_names)):
            epoch_data = selected_epoch.copy().get_data(picks=[channels]).squeeze()
            if samples < len(epoch_data):
                amplitude = epoch_data[samples]
                amplitudes.append(amplitude)
        non_stim_amplitudes = [amplitudes[i] for i in range(len(amplitudes))
                               if i not in stim_indices]
        all_amplitudes.append(non_stim_amplitudes)
    print('Done!')
    # Winsorize all amplitudes
    all_amplitudes = winsorize(np.asarray(all_amplitudes), limits=[0.01, 0.01])
    # Normalize all amplitudes
    all_amplitudes = pd.DataFrame(all_amplitudes)
    if THRESH:
        for i, value in enumerate(all_amplitudes):
            for j, subvalue in enumerate(all_amplitudes[i]):
                if all_amplitudes[i][j] < thresh and all_amplitudes[i][j] > -thresh:
                    all_amplitudes[i][j] = 0.0
                else:
                    all_amplitudes[i][j] = all_amplitudes[i][j]
    all_amplitudes_norm = all_amplitudes / all_amplitudes.abs().max().max()

    # Map all amplitudes to colors
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    amp_normed = norm(all_amplitudes_norm.values)
    rgba = colormaps.get_cmap("seismic")
    all_colors = rgba(amp_normed)
    stim_color = np.array([1, 1, 0, 1]) # yellow

    for samples in range(len(selected_epoch.times)):
        # Progress bar
        j = (samples + 1) / len(selected_epoch.times)
        sys.stdout.write('\r')
        sys.stdout.write("Plotting samples: [%-20s] %d%% " % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        sensor_colors = all_colors[samples, :, :]
        for stim_index in stim_indices:
            sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

        # plot the brain with the electrode colors for amplitude
        brain.add_sensors(raw_ecog.info,
                        trans="fsaverage",
                        ecog = True,
                        sensor_colors=sensor_colors,
                        verbose = False)
        im = brain.screenshot(mode="rgb")
        brain.remove_sensors(kind=None)
        
        # Plot the CCEP amplitude over time
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        ccep_figure = plt.figure(figsize=(10, 10))
        axis_0 = plt.subplot(gs[0])
        axis_0.imshow(im)
        axis_1 = plt.subplot(gs[1])
        selected_epoch.plot(exclude=[stim_pair[0], stim_pair[1]], axes=axis_1, show=False)
        axis_1.axvline(x=samples*(1/2048)+tmin)
        axis_1.set_ylim(-500, 500)
        if THRESH:
            axis_1.axhline(y=thresh * 1000000, color='r', linestyle='--')
            axis_1.axhline(y=-thresh * 1000000, color='r', linestyle='--')

        io_buf = io.BytesIO()
        ccep_figure.savefig(io_buf, format='raw')
        io_buf.seek(0)
        fig_data = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(ccep_figure.bbox.bounds[3]),
                                      int(ccep_figure.bbox.bounds[2]), -1))
        io_buf.close()

        pil_im = Image.fromarray(fig_data)
        images.append(pil_im)
        plt.close(ccep_figure)
    brain.close()
    
    print('Done!')
    print('Creating GIF...')
    image_one = images[0]
    os.chdir(r"C:\Users\jjbte\OneDrive\Documenten\TM3\Afstuderen\CCEP_GIF")
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_brain_annot.gif", format="GIF",
                   append_images=images[1:],
                   save_all=True, duration=100, loop=0)
    print('GIF created!')
    del images, image_one, pil_im, im, all_colors, all_amplitudes

def plot_ccep_latency(latencies, raw_ecog):
    """
    Function to plot the CCEP latency.
    """
    ## Define the colormap
    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]

    # scale latency values to be between 0 and 1, then map to colors
    non_stim_latencies = np.delete(latencies, stim_indices)
    # Winsorize the data
    non_stim_latencies = winsorize(non_stim_latencies, limits=[0.01, 0.01])
    # convert to dataframe
    non_stim_latencies = pd.DataFrame(non_stim_latencies)

    non_stim_latencies_norm = 2 * (non_stim_latencies - non_stim_latencies.min()) \
                                    / (non_stim_latencies.max() - non_stim_latencies.min()) - 1
    rgba = colormaps.get_cmap("Blues_r")
    sensor_colors = np.array(non_stim_latencies_norm.map(rgba).values.tolist(), float)
    stim_color = np.array([1, 1, 0, 1]) # yellow

    # Insert stimulation pair color at EPOCH_INDEX
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

    ## Plot the brain with the electrode colors for latency
    # plot the brain with the electrode colors for latency
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage",
                  hemi="both",
                  surf="pial",
                  cortex="grey",
                  subjects_dir=subjects_dir,
                  background="white",
                  interaction="terrain",
                  show=True)
    brain.add_annotation("aparc.a2009s",
                         borders=False,
                         alpha=0.8)
    brain.add_sensors(raw_ecog.info,
                      trans="fsaverage",
                      ecog = True,
                      sensor_colors=sensor_colors)
    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

def plot_ccep_gamma(evoked, raw_ecog):
    """ 
    Function to plot the gamma power.
    """
    # Make a copy of the epoch to avoid modifying the original
    gamma_power_t = (evoked[STIM_PAIR].copy().filter(30, 90).apply_hilbert(envelope=True))
    gamma_power_at_15s = gamma_power_t.to_data_frame(index="time").loc[0]

    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names) 
                    if stim_pair[0] in s or stim_pair[1] in s]

    # scale values to be between 0 and 1, then map to colors
    non_stim_gamma = gamma_power_at_15s.drop(stim_pair)
    non_stim_gamma_norm = 2 * (non_stim_gamma - non_stim_gamma.min()) \
                                    / (non_stim_gamma.max() - non_stim_gamma.min()) - 1
    rgba = colormaps.get_cmap("viridis")
    sensor_colors = np.array(non_stim_gamma_norm.map(rgba).tolist(), float)
    stim_color = np.array([1, 0, 0, 1]) # Red

    # Insert stimulation pair color at EPOCH_INDEX
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

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

def plot_mov_gamma(evoked, raw_ecog):
    '''
    Function to plot the gamma power over time.
    '''
    # Extract CCEP gamma power
    tmin = -0.1
    tmax = 0.2
    selected_epoch = evoked[STIM_PAIR].copy()
    selected_epoch = selected_epoch.crop(tmin = tmin, tmax = tmax)
    epoch_gamma = (selected_epoch.copy().filter(30, 90).apply_hilbert(envelope=True))
    all_gamma = []
    images = []

    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]

    for samples in range(len(epoch_gamma.times)):
        # Progress bar
        j = (samples + 1) / len(selected_epoch.times)
        sys.stdout.write('\r')
        sys.stdout.write("Calculating sample power: [%-20s] %d%% " % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        # Determine the amplitude of the CCEP response
        gammas = []
        for channels in range(len(epoch_gamma.ch_names)):
            gamma_data = epoch_gamma.copy().get_data(picks=[channels])
            if samples < len(gamma_data):
                gamma = gamma_data[samples]
                gammas.append(gamma)
        non_stim_gamma = [gammas[i] for i in range(len(gammas)) if i not in stim_indices]
        all_gamma.extend(non_stim_gamma)
    print('Done!')

    # Normalize all gamma power
    all_gamma = pd.DataFrame(all_gamma)
    all_gamma_norm = 2 * (all_gamma - all_gamma.min()) \
                          / (all_gamma.max() - all_gamma.min()) - 1
    # Map all gamma to colors
    rgba = colormaps.get_cmap("viridis")
    all_colors = np.array(all_gamma_norm.map(rgba).values.tolist(),float)
    stim_color = np.array([1, 0, 0, 1]) # Red

    for samples in range(len(epoch_gamma.times)):
        # Progress bar
        j = (samples + 1) / len(epoch_gamma.times)
        sys.stdout.write('\r')
        sys.stdout.write("Plotting samples: [%-20s] %d%% " % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        sensor_colors = all_colors[:, samples, :]
        for stim_index in stim_indices:
            sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

        # plot the brain with the electrode colors for gamma power
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
        mne.viz.close_3d_figure(fig)

        # Plot the CCEP gamma over time
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        ccep_figure = plt.figure(figsize=(10, 10))
        axis_0 = plt.subplot(gs[0])
        axis_0.imshow(im)
        axis_1 = plt.subplot(gs[1])
        epoch_gamma.plot(exclude=[stim_pair[0], stim_pair[1]], axes=axis_1, show=False)
        axis_1.axvline(x=samples*(1/2048)+tmin)

        io_buf = io.BytesIO()
        ccep_figure.savefig(io_buf, format='raw')
        io_buf.seek(0)
        fig_data = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(ccep_figure.bbox.bounds[3]),
                                      int(ccep_figure.bbox.bounds[2]), -1))
        io_buf.close()

        pil_im = Image.fromarray(fig_data)
        images.append(pil_im)
        plt.close(ccep_figure)

    print('Done!')
    print('Creating GIF...')
    image_one = images[0]
    os.chdir(r"C:\Users\jjbte\OneDrive\Documenten\TM3\Afstuderen\CCEP_GIF")
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_GAMMA.gif",
                   format="GIF", append_images=images[1:],
                   save_all=True, duration=100, loop=0)
    print('GIF created!')
    del images, image_one, pil_im, im, xy, fig, all_colors, all_gamma

if __name__ == "__main__":
    st = time.time()
    main()
    print("CCEP analysis completed!")
    et = time.time()
    res = et-st
    res_min = res // 60
    print(f"Time taken: {res_min} minutes")
# End of file
