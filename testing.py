# Import necessary libraries
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the ECoG dataset
# Replace 'path_to_your_ECoG_data.vhdr' with the actual path to your BrainVision header file
raw = mne.io.read_raw_brainvision('D:\CCEP_Data_Utrecht\sub-ccepAgeUMCU01\ses-1\ieeg\sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.vhdr', preload=True)

# Apply a band-pass filter
raw.filter(l_freq=0.1, h_freq=70)

# Plot the raw data
raw.plot()

# Perform a Fast Fourier Transform (FFT) to analyze frequency components
psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1., fmax=50.)

# Convert power spectral densities (PSDs) to dB
psds_db = 10 * np.log10(psds)

# Plot the PSD
plt.figure(figsize=(7, 4))
plt.plot(freqs, psds_db.mean(0).T, color='k', scalex=True, scaley=True)
plt.title('PSD (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.show()