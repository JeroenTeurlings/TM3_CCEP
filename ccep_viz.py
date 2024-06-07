"""

"""
# Import necessary libraries
import os
import io
import time
import sys
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
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

# Import custom modules
import ccep_connectivity

# Global settings for plotting
mne.viz.set_3d_backend('pyvista')

# Get filename for subject
Tk().withdraw()
filename = askopenfilename()
filename = os.path.basename(filename)

# Split the filename to get the subject name, session, task, and run
filename = filename.split('_')
SUBJECT = filename[0]
SESSION = filename[1]
TASK = filename[2]
RUN = filename[3]
print(f'Subject: {SUBJECT}, Session: {SESSION}, Task: {TASK}, Run: {RUN}')

# paths to mne datasets - FreeSurfer subject
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# Atlas transparency
ALPHA = 0

## Binary switches for plotting
# Basic plots
PLOT_RAW = False
PLOT_ELECTRODES = True
PLOT_ELECTRODES_GIF = False

# Epoch plots. Provide specific STIM_PAIR to plot!
STIM_PAIR = "FC01-FC02"
PLOT_EPOCHS = False
PLOT_PEAKS = False
BINARIZE_PEAKS = False
PLOT_CCEP_AMPLITUDE = False
PLOT_CCEP_LATENCY = False
PLOT_CCEP_GAMMA = False

# Animation plots. Computationally heavy!
PLOT_MOV_AMPLITUDE = False
PLOT_MOV_GAMMA = False

# Graph analysis
TOTAL_ACTIVITY = True

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
    amplitudes, latencies = find_ccep_peaks(evoked, electrodes)
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
    # Display the total significant electrode activity on a brain
    if TOTAL_ACTIVITY:
        total_activity(raw_ecog, electrodes)

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
    global SFREQ
    SFREQ = raw.info['sfreq']
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
    # Filter out the good ECoG channels in channels.tsv
    ecog_channels = channels[channels['type'] == 'ECOG']['name'].tolist()
    good_channels = channels[channels['status'] == 'good']['name'].tolist()

    # Filter out the good channels from ecog_channels
    ecog_channels = [channel for channel in ecog_channels if channel in good_channels]

    # Pick ECoG channels in raw data
    orig_raw = raw.copy()
    raw_ecog = raw.pick(ecog_channels)

    # Set channel types to ECoG
    raw_ecog.set_channel_types({ch: 'ecog' for ch in ecog_channels})

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
    brain_electrodes = mne.viz.get_brain_class()
    brain = brain_electrodes("fsaverage",
                hemi="both",
                surf="pial",
                cortex="grey",
                subjects_dir=subjects_dir,
                background="white",
                interaction="terrain",
                show=True)
    brain.add_annotation("aparc.a2009s",
                        borders=False,
                        alpha=ALPHA)
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

        os.chdir("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/CCEP_GIF")
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
    evoked = {site: epoch[site].average() for site in event_id.keys() if len(epoch[site]) > 0}

    # apply baseline correction for every evoked in evoked
    for site in evoked:
        evoked[site].apply_baseline(baseline = (None, -0.1), verbose = False)
    return evoked, epoch

def plot_epochs(epoch):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epoch[STIM_PAIR].plot(block = True, title="Epochs")
    # epoch[STIM_PAIR].plot_image(title="Epochs", picks='FC09', ts_args={"ylim":{"ecog":[-500, 500]}})

def find_ccep_peaks(evoked, electrodes):
    """
    Function to find the peaks of the CCEP response.
    """
    # Extract CCEP amplitudes
    PEAK_SD = 3.4
    for stim_pair in evoked:
        selected_epoch = evoked[stim_pair].copy()
        amplitudes = []
        latencies = []
        significant_electrodes = []

        for channels, e in enumerate(selected_epoch.ch_names):
            epoch_sd = np.std(selected_epoch.get_data(picks=[channels], tmin=-2, tmax=-0.1))
            if epoch_sd < 50/1000000:
                epoch_sd = 50/1000000
            epoch_data = selected_epoch.get_data(picks=[channels],
                                                tmin=0.009,
                                                tmax=0.100).squeeze()*-1 # Invert data for N1 peak
            amplitudes_index, _ = scipy.signal.find_peaks(epoch_data,
                                                        distance= 200, # To prevent multiple peaks
                                                        height = PEAK_SD * epoch_sd,
                                                        prominence = 20/1000000,)
            # If index is empty (no peaks are found), set amplitude to 0
            if amplitudes_index.size == 0:
                amplitude = np.array([0])
                latency = np.array([0])
            else:
                amplitude = epoch_data[amplitudes_index]
                # add 2 seconds to account for measurement window
                latency = selected_epoch.times[amplitudes_index] + 2

            if amplitude > 2.6 * epoch_sd and TOTAL_ACTIVITY:
                channel_name = selected_epoch.ch_names[channels]
                stim_0 = stim_pair.split('-')[0]
                x_stim = electrodes[electrodes['name'] == stim_0]['x'].values[0]
                y_stim = electrodes[electrodes['name'] == stim_0]['y'].values[0]
                z_stim = electrodes[electrodes['name'] == stim_0]['z'].values[0]
                destrieux_stim = electrodes[electrodes['name'] == stim_0] \
                    ['Destrieux_label'].values[0]
                x_rec = electrodes[electrodes['name'] == channel_name]['x'].values[0]
                y_rec = electrodes[electrodes['name'] == channel_name]['y'].values[0]
                z_rec = electrodes[electrodes['name'] == channel_name]['z'].values[0]
                destrieux_rec = electrodes[electrodes['name'] == channel_name] \
                    ['Destrieux_label'].values[0]
                significant_electrode = {
                    'stim_name': SUBJECT[-2:] + '_' + stim_0,
                    'xyz_stim': [x_stim, y_stim, z_stim],
                    'destrieux_stim': int(destrieux_stim),
                    'rec_name': SUBJECT[-2:] + '_' + channel_name,
                    'xyz_rec': [x_rec, y_rec, z_rec],
                    'destrieux_rec': int(destrieux_rec),
                    'amplitude': amplitude,
                    'latency': latency,
                }
                significant_electrodes.append(significant_electrode)

                stim_1 = stim_pair.split('-')[1]
                x_stim = electrodes[electrodes['name'] == stim_1]['x'].values[0]
                y_stim = electrodes[electrodes['name'] == stim_1]['y'].values[0]
                z_stim = electrodes[electrodes['name'] == stim_1]['z'].values[0]
                destrieux_stim = electrodes[electrodes['name'] == stim_1] \
                    ['Destrieux_label'].values[0]
                x_rec = electrodes[electrodes['name'] == channel_name]['x'].values[0]
                y_rec = electrodes[electrodes['name'] == channel_name]['y'].values[0]
                z_rec = electrodes[electrodes['name'] == channel_name]['z'].values[0]
                destrieux_rec = electrodes[electrodes['name'] == channel_name] \
                    ['Destrieux_label'].values[0]
                significant_electrode = {
                    'stim_name': SUBJECT[-2:] + '_' + stim_1,
                    'xyz_stim': [x_stim, y_stim, z_stim],
                    'destrieux_stim': int(destrieux_stim),
                    'rec_name': SUBJECT[-2:] + '_' + channel_name,
                    'xyz_rec': [x_rec, y_rec, z_rec],
                    'destrieux_rec': int(destrieux_rec),
                    'amplitude': amplitude,
                    'latency': latency,
                }
                significant_electrodes.append(significant_electrode)

            if PLOT_PEAKS:
                epoch_data_plot = selected_epoch.get_data(picks=[channels],
                                                tmin=-0.050,
                                                tmax=0.100).squeeze()
                time_steps = np.linspace(-0.050, 0.100, epoch_data_plot.size)
                # Calculate the number of samples
                num_samples = int((0.009 - -0.050) * SFREQ)
                amplitudes_index = amplitudes_index + num_samples
                amplitudes_times = time_steps[amplitudes_index]
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (V)')
                plt.plot(time_steps, epoch_data_plot, color='silver')
                plt.plot(time_steps[int(0.059*SFREQ):],
                        epoch_data_plot[int(0.059*SFREQ):],
                        color='black')
                plt.plot(amplitudes_times, epoch_data_plot[amplitudes_index], "x", color='r')
                plt.axhline(y = 0, linestyle = "-", color="black")
                plt.axvline(x=0, color='black', linestyle='-')
                plt.ylim(-0.0005, 0.0005)
                plt.xlim(-0.050, 0.100)
                plt.axhline(y=(PEAK_SD * epoch_sd), color='r', linestyle='--')
                plt.axhline(y=-(PEAK_SD * epoch_sd), color='r', linestyle='--')
                plt.axvline(x=0.009, color='black', linestyle='dotted')
                plt.show()
                # Print if amplitude is below threshold
                if amplitude < PEAK_SD * epoch_sd and amplitude > 0:
                    print(f'Channel {channels} has an amplitude of {amplitude}'
                        f' which is below the threshold of {epoch_sd*PEAK_SD}')
                elif amplitude > PEAK_SD * epoch_sd:
                    print('Amplitude correct')
                else:
                    print(f'Channel {channels} has no peak above the threshold')
                    continue
                # Print peak information
                print(f'Channel {channels} has a peak at {latency}'
                    f' seconds with an amplitude of {amplitude}')
            amplitudes.append(amplitude)
            latencies.append(latency)

        if TOTAL_ACTIVITY:
            # Save significant electrodes to a TSV file
            path = f"C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}"
            Path(path).mkdir(parents=True, exist_ok=True)
            os.chdir(path)
            sig_elec_df = pd.DataFrame(significant_electrodes)
            sig_elec_df.to_csv(f"{SUBJECT}_{SESSION}_{RUN}_{stim_pair}.tsv",
                            sep='\t', index=False)

        # Binarize the peaks
        if BINARIZE_PEAKS:
            amplitudes = np.where(np.array(amplitudes) > 0, 1, 0).squeeze()

    return amplitudes, latencies

def plot_ccep_amplitude(amplitudes, raw_ecog):
    """
    Function to plot the CCEP amplitude.
    """
    ## Initialize variables
    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    # Define the colormap
    rgba = colormaps.get_cmap("Reds")
    stim_color = np.array([1, 1, 0, 1]) # yellow

    ## Initialize brain
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
                        alpha=ALPHA) # Make the annotation transparent or not
    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

    ## Creating normalized amplitudes  
    amplitudes_rec = pd.DataFrame(np.delete(amplitudes, stim_indices)) # Remove stimulation pair
    amp_range = amplitudes_rec.max() - amplitudes_rec.min() # Range
    amp_low = amplitudes_rec - amplitudes_rec.min() # Low
    amplitudes_rec_norm = 2*(amp_low)/(amp_range)-1 # Normalize
    # Map amplitudes to colors
    sensor_colors = np.array(amplitudes_rec_norm.map(rgba).values.tolist(), float)
    # Insert stimulation pair color
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

    ## Add sensors to the brain
    brain.add_sensors(raw_ecog.info,
                    trans="fsaverage",
                    ecog = True,
                    sensor_colors=sensor_colors)

def plot_mov_amplitude(evoked, raw_ecog):
    """
    Function to plot the CCEP amplitude over time.
    """
    ## Initialize variables
    # Initialize brain
    brain_amplitude = mne.viz.get_brain_class()
    brain = brain_amplitude("fsaverage",
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
                            alpha=ALPHA,
                            remove_existing=True)
    # Initialize variables
    tmin = -0.1 # Start of the epoch
    tmax = 0.2 # End of the epoch
    selected_epoch = evoked[STIM_PAIR].copy()
    selected_epoch = selected_epoch.crop(tmin = tmin, tmax = tmax)
    all_amplitudes = []
    frames = []
    thresh = 2.6*50/1000000
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    stim_color = np.array([1, 1, 0, 1]) # yellow
    rgba = colormaps.get_cmap("seismic")

    ## Main loop
    for samples, e in enumerate(selected_epoch.times):
        # Progress bar
        progress_bar(samples, len(selected_epoch.times),
                    message = "Calculating sample amplitudes: ")

        amplitudes = []
        # Determine the amplitude of the CCEP response
        for channels, el in enumerate(selected_epoch.ch_names):
            epoch_data = selected_epoch.copy().get_data(picks=[channels]).squeeze()
            if samples < len(epoch_data):
                amplitude = epoch_data[samples]
                amplitudes.append(amplitude)
        non_stim_amplitudes = [amplitudes[i] for i in range(len(amplitudes))
                            if i not in stim_indices]
        all_amplitudes.append(non_stim_amplitudes)
    print('Done!')

    if BINARIZE_PEAKS:
        print('Binarizing amplitudes...')
        all_amplitudes = np.where(np.array(all_amplitudes) < -thresh, -1, 0).squeeze()
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        amp_normed = norm(all_amplitudes)
        print('Amplitudes binarized!')
    else:
        # Winsorize all amplitudes
        all_amplitudes = winsorize(np.asarray(all_amplitudes), limits=[0.01, 0.01])
        # Normalize all amplitudes
        all_amplitudes = pd.DataFrame(all_amplitudes)
        all_amplitudes = all_amplitudes / all_amplitudes.abs().max().max()
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        amp_normed = norm(all_amplitudes.values)
    all_colors = rgba(amp_normed)

    for samples, e in enumerate(selected_epoch.times):
        # Progress bar
        progress_bar(samples, len(selected_epoch.times), message = "Plotting samples: ")

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
        selected_epoch.plot(exclude=[stim_pair[0], stim_pair[1]],
                            axes=axis_1, show=False)
        axis_1.axvline(x=samples*(1/SFREQ)+tmin)
        axis_1.set_ylim(-500, 500)
        if BINARIZE_PEAKS:
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
        frames.append(pil_im)
        plt.close(ccep_figure)
    brain.close()

    print('Done!')
    print('Creating GIF...')
    image_one = frames[0]
    os.chdir("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/CCEP_GIF")
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_binarized_test.gif", format="GIF",
                append_images=frames[1:],
                save_all=True, duration=100, loop=0)
    print('GIF created!')
    del frames, image_one, pil_im, im, all_colors, all_amplitudes

def plot_ccep_latency(latencies, raw_ecog):
    """
    Function to plot the CCEP latency.
    """
    ## Initialize variables
    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    stim_color = np.array([1, 1, 0, 1]) # yellow
    rgba = colormaps.get_cmap("Blues_r")

    ## Initialize brain
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
                        alpha=ALPHA)
    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

    ## Creating normalized latencies
    # scale latency values to be between 0 and 1, then map to colors
    latencies_rec = np.delete(latencies, stim_indices) # Remove stimulation pair
    latencies_rec = winsorize(latencies_rec, limits=[0.01, 0.01]) # Winsorize
    latencies_rec = pd.DataFrame(latencies_rec) # Convert to DataFrame
    latencies_range = latencies_rec.max() - latencies_rec.min() # Range
    latencies_low = latencies_rec - latencies_rec.min() # Low
    latencies_rec_norm = 2*(latencies_low)/(latencies_range)-1 # Normalize
    # Make colors
    sensor_colors = np.array(latencies_rec_norm.map(rgba).values.tolist(), float)
    # Insert stimulation pair color at EPOCH_INDEX
    for stim_index in stim_indices:
        sensor_colors = np.insert(sensor_colors, stim_index, stim_color, axis=0)

    ## Add sensors to the brain
    brain.add_sensors(raw_ecog.info,
                    trans="fsaverage",
                    ecog = True,
                    sensor_colors=sensor_colors)

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
    ## Initialize variables
    # Initialize brain
    brain_gamma = mne.viz.get_brain_class()
    brain = brain_gamma("fsaverage",
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
                            alpha=ALPHA,
                            remove_existing=True)
    # Initialize variables
    tmin = -0.1
    tmax = 0.2
    all_gamma = []
    images = []
    selected_epoch = evoked[STIM_PAIR].copy()
    selected_epoch = selected_epoch.crop(tmin = tmin, tmax = tmax)
    epoch_gamma = (selected_epoch.copy().filter(30, 90).apply_hilbert(envelope=True))
    stim_color = np.array([1, 0, 0, 1]) # Red
    rgba = colormaps.get_cmap("viridis")
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]

    ## Main loop
    for samples, e in enumerate(epoch_gamma.times):
        # Progress bar
        progress_bar(samples, len(epoch_gamma.times),
                    message = "Calculating sample gamma power: ")

        # Determine the amplitude of the CCEP response
        gammas = []
        for channels, e1 in enumerate(epoch_gamma.ch_names):
            gamma_data = epoch_gamma.copy().get_data(picks=[channels])
            if samples < len(gamma_data):
                gamma = gamma_data[samples]
                gammas.append(gamma)
        non_stim_gamma = [gammas[i] for i, e2 in enumerate(gammas) if i not in stim_indices]
        all_gamma.extend(non_stim_gamma)
    print('Done!')

    # Normalize all gamma power
    all_gamma = pd.DataFrame(all_gamma)
    all_gamma_norm = 2 * (all_gamma - all_gamma.min()) \
                        / (all_gamma.max() - all_gamma.min()) - 1
    # Map all gamma to colors    
    all_colors = np.array(all_gamma_norm.map(rgba).values.tolist(),float)

    for samples, e in enumerate(epoch_gamma.times):
        # Progress bar
        progress_bar(samples, len(epoch_gamma.times), message = "Plotting samples: ")

        sensor_colors = all_colors[:, samples, :]
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

        # Plot the CCEP gamma over time
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        ccep_figure = plt.figure(figsize=(10, 10))
        axis_0 = plt.subplot(gs[0])
        axis_0.imshow(im)
        axis_1 = plt.subplot(gs[1])
        epoch_gamma.plot(exclude=[stim_pair[0], stim_pair[1]], axes=axis_1, show=False)
        axis_1.axvline(x=samples*(1/SFREQ)+tmin)

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
    os.chdir("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/CCEP_GIF")
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_GAMMA.gif",
                format="GIF", append_images=images[1:],
                save_all=True, duration=100, loop=0)
    print('GIF created!')
    del images, image_one, pil_im, im, all_colors, all_gamma

def total_activity(raw_ecog, electrodes):
    '''
    Display the total significant electrode activity on a brain.

    Parameters:
    raw_ecog (mne.io.Raw): The raw electrocorticography (ECoG) data.

    Returns:
    None
    '''
    os.chdir("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/TM3_CCEP")
    ## Display the total significant electrode activity on a brain
    # Run ccep_connectivity.py to get the channel_name_counts.tsv file
    ccep_connectivity.SUBJECT = SUBJECT
    ccep_connectivity.main()

    ## Initialize variables
    os.chdir(f"C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}")
    cwd = os.getcwd()
    rgba = colormaps.get_cmap("Reds")
    channel_count = pd.DataFrame(pd.read_csv(cwd +f"/output/{SUBJECT}_channel_name_counts.tsv",
                                            sep='\t'))

    # ## Initialize brain
    # brain_total = mne.viz.get_brain_class()
    # brain = brain_total("fsaverage",
    #             hemi="both",
    #             surf="pial",
    #             cortex="grey",
    #             subjects_dir=subjects_dir,
    #             background="white",
    #             interaction="terrain",
    #             show=True)
    # brain.add_annotation("aparc.a2009s",
    #                     borders=False,
    #                     alpha=ALPHA) # Make the annotation transparent or not
    # brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

    # Add xyz coordinates from electrodes to channel_count dataframe
    # channel_count = channel_count.join(electrodes.set_index('name')[['x', 'y', 'z']],
    # on='channel_name')

    # # Check if all electrodes in montage are present in the channel_count dataframe
    # for electrode in raw_ecog.ch_names:
    #     if electrode not in channel_count['channel_name'].values:
    #         x_coordinate = electrodes[electrodes['name'] == electrode]['x'].values[0]
    #         y_coordinate = electrodes[electrodes['name'] == electrode]['y'].values[0]
    #         z_coordinate = electrodes[electrodes['name'] == electrode]['z'].values[0]
    #         channel_count = pd.concat([channel_count,
    #                                 pd.DataFrame({'stim_pair': ['Unknown'],
    #                                                 'xyz_stim': [0, 0, 0],
    #                                                 'destrieux_stim': ['Unknown'],
    #                                                 'channel_name': [electrode],
    #                                                 'xyz_rec': [x_coordinate, y_coordinate, z_coordinate],
    #                                                 'destrieux_rec': ['Unknown'],
    #                                                 #'count': [0]
    #                                                 })],
    #                                 ignore_index=True)
    #         print(f"Empty electrode {electrode} appended")
    #     else:
    #         continue
    # Save appended channel_count dataframe
    path = 'C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/total_activity/'
    channel_count.to_csv(path + f"{SUBJECT}_output.tsv", sep='\t', index=False)
    # # Normalize the counts
    # channel_count['count'] = channel_count['count'] / channel_count['count'].max()
    # # Order the channel_count dataframe so it is in the same order as the raw_ecog
    # channel_count = channel_count.set_index('stim_name').reindex(raw_ecog.ch_names).reset_index()

    # # Map counts to colors
    # sensor_colors = np.array(channel_count['count'].map(rgba).values.tolist(), float)

    # ## Add sensors to the brain
    # brain.add_sensors(raw_ecog.info,
    #                 trans="fsaverage",
    #                 ecog = True,
    #                 sensor_colors=sensor_colors)

def progress_bar(samples, total_samples, message = ""):
    """
    Function to display a progress bar.
    """
    j = (samples + 1) / total_samples
    sys.stdout.write('\r')
    sys.stdout.write(message + "[%-20s] %d%% " % ('='*int(20*j), 100*j))
    sys.stdout.flush()

if __name__ == "__main__":
    st = time.time()
    main()
    print("CCEP analysis completed!")
    et = time.time()
    res = et-st
    res_min = res // 60
    print(f"Time taken: {res_min} minutes")
# End of file
