"""
This module loads BrainVision data, applies a band-pass filter, plots the raw data,
performs a Fast Fourier Transform (FFT) to analyze frequency components, 
converts power spectral densities (PSDs) to dB, and plots the PSD.
"""
# Import necessary libraries
import os
import select
import mne
from mne.epochs import Epochs
from mne.viz import plot_alignment, snapshot_brain_montage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import nibabel as nib
from nilearn import datasets, plotting, image
# from nilearn.image import new_img_like

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
    epochs, raw_ecog = preprocess(raw, channels, events, electrodes)
    # Make scatterplot of available electrode locations
    # plot_electrodes(electrodes, raw_ecog)
    # #Plot the PSD
    # plot_psd(raw)
    # #Plot the epochs
    # plot_epochs(epochs)
    # # Map the amplitude of the CCEP response
    # amplitudes = map_amplitude(epochs)
    # #Map the CCEP response on the brain
    # map_overlay(amplitudes, electrodes, raw)

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

def preprocess(raw, channels, events, electrodes):
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
            # epochs.plot_image(title=f'Averaged CCEP Response for stimpair {stim_pair}')
        except ValueError:
            print('\033[1m' + f'No epochs found for {stim_pair}' + '\033[0m')
    epochs[0].plot_image(picks = [0], title='Averaged CCEP Response')

    return epochs, raw_ecog

def plot_epochs(epochs):
    """
    Function to plot the epochs.
    """
    # Plot the epochs
    epochs[0].plot(block=True, title="Epochs")

def map_amplitude(epochs):
    """
    Function to map the amplitude of the CCEP response.
    """
    # Extract CCEP amplitudes
    selected_epoch = epochs[1].copy()
    amplitudes = selected_epoch.get_data(tmin = 0.009, tmax = 0.1).max(axis=2) - selected_epoch.get_data(tmin = 0.009, tmax = 0.1).min(axis=2)
    # amplitudes_grid = amplitudes[0, :48].reshape(6, 8)
    # plt.figure(figsize=(10, 6))
    sns.heatmap(amplitudes, cmap='viridis')
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
