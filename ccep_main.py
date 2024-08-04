"""
This script, `ccep_viz.py`, is designed for visualizing data related to 
Cortical Current Evoked Potentials (CCEP). It provides functionalities to process, analyze,
and plot CCEP data, facilitating the understanding and interpretation of cortical responses
to electrical stimulation. Key features include data loading, preprocessing, analysis routines, 
and visualization tools to generate insightful plots and graphs.

Usage:
To use this script, ensure that all dependencies are installed and that the necessary data files
are available in the specified directory. Modify the script's parameters as needed to fit your data
and analysis requirements. Run the script from in an interactive window to assure smooth execution.

Remember to fill in specific data paths, parameters, and settings as needed for your analysis.

Note:
This script is part of a larger project on studying cortical responses and is intended for research
purposes. It is not a standalone application and may require additional modifications to suit
specific use cases or data formats.

Can be used in combination with ccep_merger.py, ccep_destrieux.py and ccep_electrodes.py to
merge, process and visualize CCEP data in graphs.

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
import ccep_merger

# Global settings for plotting
mne.viz.set_3d_backend('pyvista')

# Select the patient and .vhdr file to load
Tk().withdraw()
print('Select the .vhdr file to load')
filename = askopenfilename()
filename = os.path.basename(filename)

# Split the filename to get the subject name, session, task, and run
filename = filename.split('_')
SUBJECT = filename[0]
SESSION = filename[1]
TASK = filename[2]
RUN = filename[3]
print(f'Subject: {SUBJECT}, Session: {SESSION}, Task: {TASK}, Run: {RUN}')

# Path to output folder for saving results
output_folder = Path("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Output")

# paths to mne datasets - FreeSurfer subject
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# Atlas transparency. 0 is fully transparent, 1 is fully opaque
ALPHA = 0

## Binary switches for plotting. Set to True to enable plotting.
# Basic plots
PLOT_RAW = False
PLOT_ELECTRODES = False
PLOT_ELECTRODES_GIF = False

# Epoch plots. Provide specific STIM_PAIR to plot! Lookup in events.tsv
STIM_PAIR = "T43-T44"
PLOT_EPOCHS = False
PLOT_PEAKS = False
BINARIZE_PEAKS = False
PLOT_CCEP_AMPLITUDE = False
PLOT_CCEP_LATENCY = False
PLOT_CCEP_GAMMA = False # NOTE: Not implemented yet! Not tested!

# Animation plots. Computationally heavy! Provide specific STIM_PAIR to plot!
PLOT_MOV_AMPLITUDE = True
PLOT_MOV_GAMMA = False # NOTE: Not implemented yet! Not tested!

# Graph analysis
TOTAL_ACTIVITY = False # Also turn on PLOT_PEAKS!

def main():
    """
    Main function to run the CCEP visualization pipeline.
    """
    # Load the data
    raw, electrodes, channels, events = load_data()
    # Plot the raw data
    if PLOT_RAW:
        plot_raw(raw)
    global SFREQ
    SFREQ = raw.info['sfreq']
    # Preprocess the raw data
    raw_ecog, orig_raw = preprocess(raw, channels)
    # Set the montage
    set_montage(raw_ecog, electrodes)
    # Make visualisation of available electrode locations
    if PLOT_ELECTRODES:
        plot_electrodes(raw_ecog)
    # Make evoked epochs
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
        total_activity()

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
    Function to plot the raw data in MNE viewer.
    """
    # Plot the raw data
    raw.plot(block=True, title="Raw Data")

def preprocess(raw, channels):
    """
    Preprocesses the raw data by filtering ECoG channels based on their status and type.

    This function filters out the ECoG channels specified in the channels DataFrame, 
    ensuring only those marked as 'good' are retained for analysis. It then copies the 
    original raw data, picks the filtered ECoG channels from this copy, and sets their 
    channel types explicitly to 'ecog'.

    Args:
        raw (mne.io.Raw): The raw data.
        channels (pandas.DataFrame): DataFrame containing information about the channels.
                                     It must include 'type' and 'status' columns.

    Returns:
        tuple: A tuple containing:
            - mne.io.Raw: The raw data with only the good ECoG channels selected.
            - mne.io.Raw: A copy of the original raw data before channel selection.

    Note:
        The function modifies the channel types of the selected ECoG channels in the 
        returned raw data object to 'ecog'.
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

    return raw_ecog, orig_raw

def set_montage(raw_ecog, electrodes):
    """
    Sets the montage for the raw ECoG data using electrode positions.

    This function constructs a montage for the raw ECoG data based on the positions of the
    electrodes. It converts the electrode positions from millimeters to meters and sets the
    montage to the raw data. The montage is created in the MNI Talairach coordinate frame,
    and fiducials are added from the MNI template.

    Args:
        raw_ecog (mne.io.Raw): The raw ECoG data to which the montage will be set.
        electrodes (pandas.DataFrame): A DataFrame containing the electrode positions with
                                        columns 'name' for electrode names, and 'x', 'y', 'z'
                                        for their respective Cartesian coordinates in millimeters.

    Note:
        The function modifies the `raw_ecog` object in-place by setting the montage.
        Ensure that the 'subjects_dir' variable is defined and points to the directory containing
        the subject MRI files, as it is used for adding MNI fiducials to the montage.
    """
    # Make electrode postions into a dictionary
    el_position = {row['name']: [row['x'], row['y'], row['z']] for index,
                row in electrodes.iterrows()}

    # Change position of electrodes to meters
    for key in el_position:
        el_position[key] = [i / 1000 for i in el_position[key]]

    # Set the coordinate frame of the montage
    montage = mne.channels.make_dig_montage(el_position, coord_frame='mni_tal')
    montage.add_mni_fiducials(subjects_dir)
    raw_ecog.set_montage(montage)

def plot_electrodes(raw_ecog):
    """
    Plots the electrode locations on a 3D brain model and optionally saves the visualization.

    This function initializes a 3D brain model using the specified subject's cortical surface and
    overlays the electrode locations. It also adds cortical parcellation annotations for
    visual reference. The electrodes are visualized on the brain model, and their locations are
    highlighted. If enabled, the function creates a GIF animation showing a 360-degree rotation
    of the brain model with the electrodes in place.

    Parameters:
    - brain_electrodes: A class from the MNE library used to initialize the 3D brain model.
    - brain: The 3D brain model object.
    - ALPHA: The transparency level for the cortical parcellation annotations.
    - PLOT_ELECTRODES_GIF: A boolean flag indicating whether to save the visualization as a GIF.
    - subjects_dir: The directory containing the subject MRI files.
    - raw_ecog: The raw ECoG data object containing the electrode information.

    The function adds annotations to the brain model to indicate different cortical regions,
    sets the electrode locations on the model, and adjusts the view to show the brain from
    a specified angle. If PLOT_ELECTRODES_GIF is True, it generates a GIF by rotating the brain
    model and capturing screenshots at each step.

    Note:
    - The function modifies the current working directory to save the GIF in a specified location.
    - Ensure that the 'subjects_dir' variable is correctly set to the directory
    """
    # Initialize the brain
    brain_electrodes = mne.viz.get_brain_class()
    brain = brain_electrodes("fsaverage",
                hemi="both",
                surf="pial",
                cortex="grey",
                subjects_dir=subjects_dir,
                background="white",
                interaction="terrain",
                show=True)

    # Add annotation to the brain. Fill in ALPHA value at the top!
    brain.add_annotation("aparc.a2009s",
                        borders=False,
                        alpha=ALPHA)

    # Add sensors to the brain
    brain.add_sensors(raw_ecog.info,
                    trans="fsaverage",
                    ecog = True,
                    sensor_colors=(1.0, 1.0, 1.0, 0.5))

    # Save a turning image of the brain as a GIF
    if PLOT_ELECTRODES_GIF:
        screenshots = []
        # Turn brain in steps of 1 degree and save screenshots
        for azimuth in range(0, 360, 1):
            brain.show_view(azimuth=azimuth,
                            elevation=90,
                            focalpoint="auto",
                            distance="auto")
            screenshots.append(Image.fromarray(brain.screenshot(mode="rgb", time_viewer=True)))
        # Fill in the path where the GIF should be saved
        os.chdir(output_folder)
        screenshots[0].save("Electrodes.gif", format="GIF",
                    append_images=screenshots[1:],
                    save_all=True, duration=50, loop=0)

    brain.show_view(azimuth=180, elevation=90, focalpoint="auto", distance="auto")

def make_epochs(raw_ecog, events):
    """
    Extracts CCEP events from the given events dataframe and creates epochs for each event,
    then averages them to obtain evoked responses.

    This function processes a dataframe containing event information to extract
    CCEP (Cortical Current Evoked Potential) events. It then segments the raw ECoG data into epochs
    based on these events and averages the epochs to obtain evoked responses for each unique
    electrical stimulation site. The function also applies baseline correction to each response.

    Parameters:
    - raw_ecog (mne.io.Raw): The raw ECoG data.
    - events (pandas.DataFrame): A dataframe containing information about the events, including
    columns for 'sample_start', 'trial_type', and 'electrical_stimulation_site'.

    Returns:
    - tuple: A tuple containing:
        - dict: A dictionary of evoked responses, keyed by electrical stimulation site.
        - mne.Epochs: The Epochs object containing all the epochs created for CCEP events.

    Raises:
    - ValueError: If no epochs are found for a specific stimulation pair, indicating that the
    events dataframe might not contain any valid CCEP events or the conditions are not met.

    The function first filters the events dataframe to retain only those rows corresponding to
    electrical stimulation events. It then maps each unique electrical stimulation site to an
    integer ID and creates an array of event markers. These markers are used to segment the
    raw ECoG data into epochs. Finally, it averages the epochs for each stimulation site to obtain
    evoked responses and applies baseline correction to each.
    """
    # Extract CCEP events from events.tsv
    events_ccep = events.loc[events['trial_type'] == 'electrical_stimulation',
                             ['sample_start', 'electrical_stimulation_site']]

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
    Function to plot and visualize the epochs.
    """
    # Plot the epochs
    epoch[STIM_PAIR].plot(block = True, title="Epochs")

def find_ccep_peaks(evoked, electrodes):
    """
    Function to find the peaks of the CCEP response.
    """
    # Specify the peak detection parameters and initialize variables
    PEAK_SD = 3.4
    total_stim_pairs = len(evoked.keys()) if TOTAL_ACTIVITY else 1
    stim_pair_counter = 0  # Initialize a counter for stimulation pairs

    # If TOTAL_ACTIVITY is True, loop through all stimulation pairs
    # Otherwise, just one stimulation pair is processed. Specify the pair in STIM_PAIR.
    if TOTAL_ACTIVITY:
        stim_pairs = evoked.keys()
        significant_electrodes = []
    else:
        stim_pairs = [STIM_PAIR]

    # Loop through each stimulation pair. Just one pair if TOTAL_ACTIVITY is False.
    for stim_pair in stim_pairs:
        stim_pair_counter += 1  # Increment the stim_pair_counter for each stim_pair processed
        selected_epoch = evoked[stim_pair].copy()
        amplitudes = []
        latencies = []
        labels = []
        significant_electrodes = []

        # Loop through each channel in the epoch
        for channels, e in enumerate(selected_epoch.ch_names):
            # Calculate the standard deviation of the epoch data
            epoch_sd = np.std(selected_epoch.get_data(picks=[channels], tmin=-2, tmax=-0.1))
            # Ensure a minimum standard deviation of 50 uV
            if epoch_sd < 50/1000000:
                epoch_sd = 50/1000000
            epoch_data = selected_epoch.get_data(picks=[channels],
                                                tmin=0.009,
                                                tmax=0.100).squeeze()*-1 # Invert data for N1 peak
            # Find the peaks in the epoch data. Peak detection parameters by Van Blooijs et al. (2023)
            amplitudes_index, _ = scipy.signal.find_peaks(epoch_data,
                                                        distance= 200, # To prevent multiple peaks
                                                        height = PEAK_SD * epoch_sd,
                                                        prominence = 20/1000000,)

            # If index is empty (no peaks are found), set amplitude and latency to 0
            if amplitudes_index.size == 0:
                amplitude = np.array([0])
                latency = np.array([0])
            else:
                amplitude = epoch_data[amplitudes_index]
                # add 2.009 seconds to account for measurement window
                latency = selected_epoch.times[amplitudes_index] + 2.009

            # Label the peaks. Visual check of all peaks. Only if TOTAL_ACTIVITY and PLOT_PEAKS are True
            if amplitude > PEAK_SD * epoch_sd and TOTAL_ACTIVITY and PLOT_PEAKS:
                channel_name = selected_epoch.ch_names[channels]
                epoch_data_plot = selected_epoch.get_data(picks=[channels],
                                                tmin=-0.050,
                                                tmax=0.100).squeeze()
                time_steps = np.linspace(-0.050, 0.100, epoch_data_plot.size)
                # Calculate the number of samples
                num_samples = int((0.009 - -0.050) * SFREQ)
                amplitudes_index = amplitudes_index + num_samples
                amplitudes_times = time_steps[amplitudes_index]
                # Plot the peaks
                stim_pair_percentage = ((stim_pair_counter-1) / total_stim_pairs) * 100
                print(f"Stim Pair Progress: {stim_pair_percentage:.2f}% - Plotting signal...")
                print('Label 0 (clean) or 1 (artefact). Or Q to save and exit')
                fig = plt.figure()
                # Add a key press event to label the peaks. Function is defined in PeakLabeler class
                labeler = PeakLabeler()
                fig.canvas.mpl_connect('key_press_event', labeler.press)
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (V)')
                plt.plot(time_steps, epoch_data_plot, color='silver')
                plt.plot(time_steps[int(0.059*SFREQ):],
                        epoch_data_plot[int(0.059*SFREQ):],
                        color='black')
                plt.plot(amplitudes_times, epoch_data_plot[amplitudes_index], "x", color='r')
                plt.axhline(y = 0, linestyle = "-", color="black")
                plt.axvline(x=0, color='black', linestyle='-')
                plt.ylim(-0.003, 0.003)
                plt.xlim(-0.050, 0.100)
                plt.axhline(y=-(PEAK_SD * epoch_sd), color='r', linestyle='--')
                plt.axvline(x=0.009, color='black', linestyle='dotted')
                plt.show(block=True)

                # Print if amplitude is below threshold
                if amplitude < PEAK_SD * epoch_sd and amplitude > 0:
                    print(f'Channel {channel_name} has an amplitude of {amplitude}'
                        f' which is below the threshold of {epoch_sd*PEAK_SD}')
                elif amplitude > PEAK_SD * epoch_sd:
                    print('Amplitude correct')
                else:
                    print(f'Channel {channel_name} has no peak above the threshold')
                    continue
                # Print peak information
                print(f'Channel {channel_name} has a peak at {latency}'
                    f' seconds with an amplitude of {amplitude}')

                # Save the labeled peaks for further analysis
                # Creates a TSV file with the labeled peaks and its information
                # Stimulation pair is seen two individual stimulations.
                channel_name = selected_epoch.ch_names[channels]
                stim_0 = stim_pair.split('-', maxsplit=1)[0]
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
                    'label': labeler.label_peak,
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
                    'label': labeler.label_peak,
                }
                significant_electrodes.append(significant_electrode)
                labels.append(labeler.label_peak)

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

        # Binarize the peaks. If enabled, set amplitudes to 1 if above threshold, 0 otherwise
        if BINARIZE_PEAKS:
            amplitudes = np.where(np.array(amplitudes) > 0, 1, 0).squeeze()

    return amplitudes, latencies

def plot_ccep_amplitude(amplitudes, raw_ecog):
    """
    Function to plot the CCEP amplitudes per electrode. The amplitudes are color-coded
    and displayed on a 3D brain model. The stimulation pair is highlighted in yellow.
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
    brain_amplitude = mne.viz.get_brain_class()
    brain = brain_amplitude("fsaverage",
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
    Function to plot the CCEP amplitude over time. The amplitudes are color-coded
    and displayed on a 3D brain model. The stimulation pair is highlighted in yellow.
    This animation is saved as a GIF file.
    """
    ## Initialize variables
    # Initialize brain
    brain_amplitude_mov = mne.viz.get_brain_class()
    brain = brain_amplitude_mov("fsaverage",
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
        # Determine the amplitude of the CCEP response for each channel
        for channels, el in enumerate(selected_epoch.ch_names):
            epoch_data = selected_epoch.copy().get_data(picks=[channels]).squeeze()
            if samples < len(epoch_data):
                amplitude = epoch_data[samples]
                amplitudes.append(amplitude)
        non_stim_amplitudes = [amplitudes[i] for i in range(len(amplitudes))
                            if i not in stim_indices]
        all_amplitudes.append(non_stim_amplitudes)
    print('Done!')

    # Binarize the peaks. If enabled, set amplitudes to 1 if above threshold, 0 otherwise
    # If not, normalize and winsorize the amplitudes. Taking the top 1% and bottom 1% of the data
    # and setting them to the 1% and 99% percentile respectively.
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

    # Loop through each sample and plot the brain with the electrode colors for amplitude
    # Plot the CCEP amplitude over time
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
        axis_1.set_ylim(-1000, 1000)
        if BINARIZE_PEAKS:
            axis_1.axhline(y=-thresh * 1000000, color='r', linestyle='--')

        # Save the figure to a buffer. This is necessary to create the GIF
        # Append the figure to the frames list
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
    # Save the frames to a GIF file
    image_one = frames[0]
    os.chdir(output_folder)
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_PROD_nonbin.gif", format="GIF",
                append_images=frames[1:],
                save_all=True, duration=100, loop=0)
    print('GIF created!')
    del frames, image_one, pil_im, im, all_colors, all_amplitudes

def plot_ccep_latency(latencies, raw_ecog):
    """
    Function to plot the CCEP latency per electrode. The latencies are color-coded
    and displayed on a 3D brain model. The stimulation pair is highlighted in yellow.
    """
    ## Initialize variables
    # Get index of stim_pair in raw_ecog
    stim_pair= STIM_PAIR.split('-')
    stim_indices = [i for i, s in enumerate(raw_ecog.ch_names)
                    if stim_pair[0] in s or stim_pair[1] in s]
    stim_color = np.array([1, 1, 0, 1]) # yellow
    rgba = colormaps.get_cmap("Blues_r")

    ## Initialize brain
    brain_latency = mne.viz.get_brain_class()
    brain = brain_latency("fsaverage",
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
    # replace 0 values with the maximum values in latencies_rec
    latencies_rec[latencies_rec == 0] = latencies_rec.max()
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
    NOTE: This function is not yet complete and not used in the corresponding thesis.
    It is not tested and may not work as intended.
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
    NOTE: This function is not yet complete and not used in the corresponding thesis.
    It is not tested and may not work as intended.
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
    os.chdir(output_folder)
    image_one.save(f"{SUBJECT}_{SESSION}_{RUN}_{STIM_PAIR}_GAMMA.gif",
                format="GIF", append_images=images[1:],
                save_all=True, duration=100, loop=0)
    print('GIF created!')
    del images, image_one, pil_im, im, all_colors, all_gamma

def total_activity():
    '''
    Calculates and displays the total significant electrode activity on a brain. 
    Necessary for further graph analysis. Merges all found peaks into one file per subject.

    Parameters:
    raw_ecog (mne.io.Raw): The raw electrocorticography (ECoG) data.
    '''
    os.chdir("C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/TM3_CCEP")
    ## Display the total significant electrode activity on a brain
    # Run ccep_merger.py to get the channel_name_counts.tsv file
    ccep_merger.SUBJECT = SUBJECT
    ccep_merger.main()

    ## Initialize variables
    os.chdir(f"C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}")
    cwd = os.getcwd()
    channel_count = pd.DataFrame(pd.read_csv(cwd +f"/output/{SUBJECT}_channel_name_counts.tsv",
                                            sep='\t'))

    # Save appended channel_count dataframe
    path = 'C:/Users/jjbte/Documents/01. Projects/TM3/Afstuderen/Significant_Electrodes/total_activity/'
    channel_count.to_csv(path + f"{SUBJECT}_output.tsv", sep='\t', index=False)

def progress_bar(samples, total_samples, message = ""):
    """
    Function to display a progress bar.
    """
    j = (samples + 1) / total_samples
    sys.stdout.write('\r')
    sys.stdout.write(message + "[%-20s] %d%% " % ('='*int(20*j), 100*j))
    sys.stdout.flush()

class PeakLabeler:
    """
    A class to label peaks in a plot interactively.

    Attributes:
        label_peak (int): The label assigned to the peak. Possible values are:
                          - 0: indicating the peak is clean.
                          - 1: indicating the peak is an artefact.
                          - 2: used to indicate the process should save and exit.

    Methods:
        press(event): Handles key press events to label peaks or exit the labeling process.
                      - '0': labels the peak as clean (0) and closes the plot.
                      - '1': labels the peak as an artefact (1) and closes the plot.
                      - 'q': sets the label to indicate save and exit (2), prints a message, and closes the plot.
    """
    def __init__(self):
        self.label_peak = None

    def press(self, event):
        print('Press', event.key)
        if event.key == '0':
            self.label_peak = 0
            plt.close()
        elif event.key == '1':
            self.label_peak = 1
            plt.close()
        elif event.key == 'q':
            self.label_peak = 2
            print('Save and exit')
            plt.close()

if __name__ == "__main__":
    st = time.time()
    main()
    print("CCEP analysis completed!")
    # Timer
    et = time.time()
    res = et-st
    res_min = int(res // 60)
    print(f"Time taken: {res_min} minutes")
# End of file
